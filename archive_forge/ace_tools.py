import os
import re
import sys
import traceback
import uuid
import json
import logging
import socket
import time
import subprocess
from pathlib import Path
from typing import Any as TypingAny, List, Dict, Optional, Literal, TypeVar, Union, cast


import pandas as pd
import requests
import pydantic
from requests.exceptions import RequestException, ConnectionError
from http.client import RemoteDisconnected

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration Defaults with fallbacks
BASE_URL = os.getenv("ACE_BASE_URL", "http://localhost") 
USER_MACHINE_PORT = os.getenv("ACE_USER_MACHINE_PORT", "8080")
FEATURE_SET = os.getenv("FEATURE_SET", "default")
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "false").lower() == "true"
MAX_RETRIES = 3
RETRY_DELAY = 2  # Base delay for exponential backoff
REQUEST_TIMEOUT = 30

# Local storage paths
LOCAL_STORAGE_DIR = Path(os.getenv("LOCAL_STORAGE_DIR", ".ace_tools"))
LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Mount path for backward compatibility
MOUNT_PATH = Path(os.getenv("MOUNT_PATH", "/mnt/data"))
MOUNT_PATH.mkdir(parents=True, exist_ok=True)

# Local server path with validation
LOCAL_SERVER_PATH = Path(os.getenv("LOCAL_SERVER_PATH", "./local_server.py"))
if not LOCAL_SERVER_PATH.exists():
    LOCAL_SERVER_PATH = Path(__file__).parent / "local_server.py"
if not LOCAL_SERVER_PATH.exists():
    raise FileNotFoundError(f"Could not find local_server.py at {LOCAL_SERVER_PATH}")

# Type definitions
T = TypeVar('T')
JsonSerializable = Union[Dict[str, TypingAny], List[TypingAny], str, int, float, bool, None]

# Core type definitions
class ObjectReference(pydantic.BaseModel):
    type: Literal["multi_kernel_manager", "kernel_manager", "client", "callbacks"]
    id: str

class MethodCall(pydantic.BaseModel):
    message_type: Literal["call_request"] = "call_request"
    object_reference: ObjectReference
    request_id: str
    method: str
    args: list[TypingAny]
    kwargs: dict[str, TypingAny]

class MethodCallException(pydantic.BaseModel):
    message_type: Literal["call_exception"] = "call_exception"
    request_id: str
    type: str
    value: str
    traceback: list[str]

class MethodCallReturnValue(pydantic.BaseModel):
    message_type: Literal["call_return_value"] = "call_return_value"
    request_id: str
    value: TypingAny = None

class MethodCallObjectReferenceReturnValue(pydantic.BaseModel):
    message_type: Literal["call_object_reference"] = "call_object_reference"
    request_id: str
    object_reference: ObjectReference

class AceException(Exception):
    pass

class UserMachineResponseTooLarge(AceException):
    pass

class ConnectionFailedError(AceException):
    pass

class ServerStartupError(AceException):
    pass

UserMachineResponse = Union[MethodCall, MethodCallException, MethodCallReturnValue, MethodCallObjectReferenceReturnValue]

_server_process = None

def start_local_server() -> None:
    """Start the local server if not already running"""
    global _server_process
    try:
        if _server_process is not None:
            # Check if process is still running
            if _server_process.poll() is None:
                return
            _server_process = None
            
        logger.info("Starting local server...")
        # Start server in background with proper Python executable
        _server_process = subprocess.Popen(
            [sys.executable, str(LOCAL_SERVER_PATH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        # Wait for server to start
        retries = 0
        while retries < 5:
            if check_connection():
                logger.info("Local server started successfully")
                return
            if _server_process.poll() is not None:
                stdout, stderr = _server_process.communicate()
                raise ServerStartupError(
                    f"Server process terminated unexpectedly.\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}"
                )
            time.sleep(2)
            retries += 1
            
        raise ServerStartupError("Failed to start local server after 5 retries")
        
    except Exception as e:
        logger.error(f"Error starting local server: {e}")
        if _server_process and _server_process.poll() is None:
            _server_process.terminate()
            _server_process.wait(timeout=5)
        _server_process = None
        raise

def check_connection() -> bool:
    """Check if connection to server is available"""
    try:
        with socket.create_connection(
            (BASE_URL.replace('http://', ''), int(USER_MACHINE_PORT)), 
            timeout=5
        ) as sock:
            return True
    except (socket.timeout, socket.error):
        return False

def ensure_server_running() -> None:
    """Ensure server is running, start local if needed"""
    if not check_connection():
        start_local_server()
        if not check_connection():
            logger.warning("Server connection failed, switching to offline mode")
            global OFFLINE_MODE
            OFFLINE_MODE = True

def save_locally(data: JsonSerializable, filename: str) -> None:
    """Save data locally as JSON if network calls fail"""
    try:
        filepath = LOCAL_STORAGE_DIR / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save locally: {e}")

def make_request(url: str, payload: dict, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
    """Make HTTP request with proper error handling"""
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response
    except RemoteDisconnected as e:
        raise ConnectionFailedError(f"Remote server disconnected: {str(e)}")
    except requests.exceptions.Timeout:
        raise ConnectionFailedError("Request timed out")
    except requests.exceptions.ConnectionError as e:
        raise ConnectionFailedError(f"Connection failed: {str(e)}")
    except Exception as e:
        raise ConnectionFailedError(f"Request failed: {str(e)}")

def call_function(
    function_name: str,
    function_args: List[TypingAny] = [],
    function_kwargs: Dict[str, TypingAny] = {}
) -> TypingAny:
    """
    Generic function to call remote methods via HTTP POST with local fallback.
    
    Args:
        function_name: Name of the remote function to call
        function_args: Positional arguments for the function
        function_kwargs: Keyword arguments for the function
        
    Returns:
        The response value from the remote function or local fallback
    """
    if OFFLINE_MODE:
        return handle_offline_fallback(function_name, function_kwargs)

    retries = 0
    last_error = None
    
    while retries < MAX_RETRIES:
        try:
            ensure_server_running()
                
            request_id = str(uuid.uuid4())
            
            payload = MethodCall(
                request_id=request_id,
                object_reference=ObjectReference(
                    type="callbacks",
                    id=os.environ.get("KERNEL_CALLBACK_ID", "")
                ),
                method=function_name,
                args=list(function_args),
                kwargs=function_kwargs
            ).model_dump()

            save_locally(payload, f"{request_id}.json")

            response = make_request(
                f"{BASE_URL}:{USER_MACHINE_PORT}/ace_tools/call_function",
                payload
            )
            return response.json().get("value")

        except ConnectionFailedError as e:
            last_error = e
            logger.warning(f"Network call failed (attempt {retries + 1}/{MAX_RETRIES}): {e}")
            retries += 1
            if retries < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (2 ** retries))  # Exponential backoff
        except Exception as e:
            last_error = e
            logger.warning(f"Unexpected error (attempt {retries + 1}/{MAX_RETRIES}): {e}")
            retries += 1
            if retries < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (2 ** retries))

    logger.error(f"All retries failed. Last error: {last_error}")
    return handle_offline_fallback(function_name, function_kwargs)

def handle_offline_fallback(function_name: str, function_kwargs: Dict[str, TypingAny]) -> TypingAny:
    """Handle offline/failure cases with appropriate fallbacks"""
    if function_name == "display_dataframe_to_user":
        df = function_kwargs.get("dataframe", pd.DataFrame())
        try:
            # Save locally
            if "path" in function_kwargs:
                df.to_csv(function_kwargs["path"], index=False)
            return df.head()
        except Exception as e:
            logger.warning(f"Failed to save dataframe: {e}")
            return pd.DataFrame().head()
            
    elif function_name == "display_chart_to_user":
        try:
            # Save chart metadata
            save_locally(function_kwargs, f"chart_{uuid.uuid4()}.json")
        except Exception as e:
            logger.warning(f"Failed to save chart metadata: {e}")
        return None
        
    elif function_name == "log_matplotlib_img_fallback":
        try:
            logger.info(f"Matplotlib fallback: {function_kwargs.get('reason', 'Unknown')}")
        except Exception as e:
            logger.warning(f"Failed to log matplotlib fallback: {e}")
        return None
        
    return None

def log_exception(
    message: str,
    exception_id: Optional[str] = None,
    func_name: Optional[str] = None,
    args: Optional[List[TypingAny]] = None, 
    kwargs: Optional[Dict[str, TypingAny]] = None,
) -> None:
    """
    Logs exceptions with local fallback.
    
    Args:
        message: Error message to log
        exception_id: Unique identifier for the exception
        func_name: Name of the function where exception occurred
        args: Function arguments at time of exception
        kwargs: Function keyword arguments at time of exception
    """
    try:
        exception_id = exception_id or str(uuid.uuid4())
        exc_type, exc_value, _exc_traceback = sys.exc_info()
        
        if exc_type is None:
            exc_type = type(None)
            
        traceback_str = "".join(traceback.format_exc())

        # Handle argument serialization failures gracefully
        args_str = None
        kwargs_str = None
        
        if args:
            try:
                args_str = str(args)
            except:
                args_str = "failed_to_serialize"
                
        if kwargs:
            try:
                kwargs_str = str(kwargs)
            except:
                kwargs_str = "failed_to_serialize"
        
        error_data = {
            "message": message,
            "exception": {
                "id": exception_id,
                "type": exc_type.__name__,
                "value": str(exc_value),
                "traceback": traceback_str,
            },
            "orig_func_name": func_name,
            "orig_func_args": args_str,
            "orig_func_kwargs": kwargs_str,
            "timestamp": time.time()
        }

        # Try network logging first
        if not OFFLINE_MODE:
            try:
                ensure_server_running()
                make_request(
                    f"{BASE_URL}:{USER_MACHINE_PORT}/ace_tools/log_exception",
                    error_data
                )
                return
            except ConnectionFailedError:
                pass

        # Local fallback logging
        logger.error(message)
        save_locally(error_data, f"error_{exception_id}.json")
        
    except Exception as e:
        # Absolute last resort
        print(f"Failed to log exception: {str(e)}")

# Core Functions
def display_dataframe_to_user(
    name: str,
    dataframe: pd.DataFrame,
    file_path: str = ".",
) -> pd.DataFrame:
    """
    Displays a dataframe to the user with robust local fallback.
    
    Args:
        name: Display name for the dataframe
        dataframe: Pandas DataFrame to display
        file_path: Path to save CSV file, defaults to current directory
        
    Returns:
        First few rows of the dataframe
    """
    if FEATURE_SET == "chatgpt-research":
        return dataframe.head()

    try:
        sanitized_name = re.sub(r"[^a-zA-Z0-9_\\-]", "_", name)
        
        # Support both new and legacy paths
        full_path = Path(file_path) / f"{sanitized_name}.csv"
        legacy_path = MOUNT_PATH / f"{sanitized_name}.csv"

        # Always try to save locally first
        try:
            # Handle index appropriately
            if isinstance(dataframe.index, pd.RangeIndex):
                dataframe.to_csv(full_path, index=False)
                dataframe.to_csv(legacy_path, index=False)
            else:
                dataframe.to_csv(full_path)
                dataframe.to_csv(legacy_path)
        except Exception as e:
            logger.warning(f"Failed to save CSV: {e}")
            # Try alternate location
            try:
                alt_path = LOCAL_STORAGE_DIR / f"{sanitized_name}.csv"
                dataframe.to_csv(alt_path, index=False)
            except Exception as e:
                logger.warning(f"Failed to save CSV to alternate location: {e}")

        if not OFFLINE_MODE:
            ensure_server_running()
            call_function(
                "display_dataframe_to_user",
                [],
                {"path": str(full_path), "title": name}
            )
        
        return dataframe.head()
        
    except Exception as e:
        logger.error(f"Error displaying dataframe: {e}")
        return dataframe.head()

def display_chart_to_user(
    path: str,
    title: str, 
    chart_type: str,
    metadata: Optional[Dict[str, TypingAny]] = None,
) -> None:
    """
    Displays a chart to the user with local fallback.
    
    Args:
        path: Path to the chart file
        title: Title of the chart
        chart_type: Type of chart to display
        metadata: Additional metadata for the chart
    """
    if FEATURE_SET == "chatgpt-research":
        return

    try:
        # Save chart metadata locally first
        chart_data = {
            "path": path,
            "title": title,
            "chart_type": chart_type,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        save_locally(chart_data, f"chart_{uuid.uuid4()}.json")

        if not OFFLINE_MODE:
            ensure_server_running()
            call_function(
                "display_chart_to_user",
                [],
                chart_data
            )
    except Exception as e:
        logger.error(f"Error displaying chart: {e}")

def log_matplotlib_img_fallback(
    reason: str,
    metadata: Optional[Dict[str, TypingAny]] = None
) -> None:
    """
    Logs a fallback for matplotlib images with local storage.
    
    Args:
        reason: Reason for the fallback
        metadata: Additional metadata about the fallback
    """
    try:
        fallback_data = {
            "reason": reason,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        # Always log locally
        logger.info(f"Matplotlib fallback: {reason}")
        save_locally(fallback_data, f"matplotlib_fallback_{uuid.uuid4()}.json")

        if not OFFLINE_MODE:
            ensure_server_running()
            call_function(
                "log_matplotlib_img_fallback",
                [],
                fallback_data
            )
    except Exception as e:
        logger.error(f"Error logging matplotlib fallback: {e}")

def display_matplotlib_image_to_user(
    title: str,
    reason: str,
    exception_ids: List[str],
) -> None:
    """
    Displays a matplotlib image to the user.
    
    Args:
        title: Title of the image
        reason: Reason for displaying the image
        exception_ids: List of related exception IDs
    """
    try:
        if not OFFLINE_MODE:
            ensure_server_running()
            call_function(
                "display_matplotlib_image_to_user",
                [],
                {
                    "title": title,
                    "reason": reason,
                    "exception_ids": exception_ids,
                    "timestamp": time.time()
                }
            )
    except Exception as e:
        logger.error(f"Error displaying matplotlib image: {e}")

def main():
    """Comprehensive test suite to validate all functionality"""
    logger.info("Starting comprehensive validation of ace_tools...")
    
    # Test configuration and environment
    logger.info("Testing configuration...")
    assert MOUNT_PATH.exists(), "Mount path not created"
    assert LOCAL_STORAGE_DIR.exists(), "Local storage directory not created"
    
    # Test server connectivity
    logger.info("Testing server connectivity...")
    ensure_server_running()
    assert check_connection(), "Server connection failed"
    
    # Test local storage
    logger.info("Testing local storage...")
    test_data = {"test": "data"}
    save_locally(test_data, "test.json")
    assert (LOCAL_STORAGE_DIR / "test.json").exists(), "Local storage failed"
    
    # Test exception handling
    logger.info("Testing exception handling...")
    try:
        raise ValueError("Test exception")
    except Exception:
        log_exception("Test exception", "test-id")
    assert (LOCAL_STORAGE_DIR / "error_test-id.json").exists(), "Exception logging failed"
    
    # Test DataFrame handling
    logger.info("Testing DataFrame handling...")
    test_df = pd.DataFrame({"test": [1, 2, 3]})
    result = display_dataframe_to_user("test_df", test_df)
    assert isinstance(result, pd.DataFrame), "DataFrame display failed"
    
    # Test chart display
    logger.info("Testing chart display...")
    display_chart_to_user("test.png", "Test Chart", "line")
    
    # Test matplotlib fallback
    logger.info("Testing matplotlib fallback...")
    log_matplotlib_img_fallback("test fallback")
    
    # Test offline mode
    logger.info("Testing offline mode...")
    global OFFLINE_MODE
    original_mode = OFFLINE_MODE
    OFFLINE_MODE = True
    try:
        result = call_function("test_function", [], {})
        assert result is None, "Offline fallback failed"
    finally:
        OFFLINE_MODE = original_mode
    
    # Test error conditions
    logger.info("Testing error conditions...")
    try:
        make_request("invalid_url", {})
    except ConnectionFailedError:
        pass
    else:
        raise AssertionError("Error handling failed")
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Test suite failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        if _server_process and _server_process.poll() is None:
            _server_process.terminate()
            _server_process.wait(timeout=5)
