import os
import time
import psutil
import queue
import threading
import tempfile
import pickle
import logging
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import torch
import tensorflow as tf
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    AutoTokenizer
)
from accelerate import disk_offload

# Initialize root logger with comprehensive formatting and styling
# Symbols: ⚙️=Config, ℹ️=Info, ⚠️=Warning, ❌=Error, 🔧=Debug, 💾=Data, 🔍=Trace, ⚡=Processing, 🔒=Security, 🌐=Network, ⏱️=Performance
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Console handler with advanced color and formatting support
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
# Rich color scheme with ANSI escape codes and styles
COLORS = {
    'DEBUG': '\033[38;5;39m',      # Bright blue
    'INFO': '\033[38;5;82m',       # Bright green  
    'WARNING': '\033[38;5;214m',   # Bright orange
    'ERROR': '\033[38;5;196m',     # Bright red
    'CRITICAL': '\033[48;5;196m\033[38;5;231m',  # Red bg + white text
    'TRACE': '\033[38;5;147m',     # Light purple
    'DATA': '\033[38;5;226m',      # Yellow
    'PROCESS': '\033[38;5;123m',   # Cyan
    'SECURITY': '\033[38;5;162m',  # Magenta
    'NETWORK': '\033[38;5;45m',    # Sky blue
    'PERF': '\033[38;5;208m',      # Orange
    'RESET': '\033[0m',            # Reset all formatting
    'BOLD': '\033[1m',             # Bold text
    'DIM': '\033[2m',              # Dimmed text
    'ITALIC': '\033[3m',           # Italic text
    'UNDERLINE': '\033[4m',        # Underlined text
    'BLINK': '\033[5m',            # Blinking text
    'REVERSE': '\033[7m',          # Reversed colors
    'STRIKE': '\033[9m'            # Strikethrough
}
class EnhancedColoredFormatter(logging.Formatter):
    """Advanced formatter with rich styling, technical details and metrics"""  
    SYMBOLS = {
        'DEBUG': '[DEBUG]',
        'INFO': '[INFO]',
        'WARNING': '[WARN]',
        'ERROR': '[ERROR]',
        'CRITICAL': '[CRIT]',
        'TRACE': '[TRACE]',
        'DATA': '[DATA]',
        'PROCESS': '[PROC]',
        'SECURITY': '[SEC]',
        'NETWORK': '[NET]',
        'PERF': '[PERF]'
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
    
    def format(self, record):
        # System metrics
        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        # Enhanced technical details
        record.memory_rss = f"{memory.rss / 1024 / 1024:.2f}MB"
        record.memory_vms = f"{memory.vms / 1024 / 1024:.2f}MB"
        record.cpu_usage = f"{cpu_percent:.1f}%"
        record.thread_id = threading.get_ident()
        record.process_name = process.name()
        record.uptime = f"{time.time() - self.start_time:.1f}s"
        record.memory_percent = f"{process.memory_percent():.1f}%"
        
        # Style the level name with color, symbol and padding
        level_symbol = self.SYMBOLS.get(record.levelname, '')
        level_color = COLORS.get(record.levelname, COLORS['RESET'])
        
        # Format level name with consistent width and styling
        record.levelname = (
            f"{level_color}{COLORS['BOLD']}{level_symbol} "
            f"{record.levelname:<8}{COLORS['RESET']}"
        )
        
        # Format module/function context with file and line info
        record.name = (
            f"{COLORS['DIM']}{record.name:<25} "
            f"({record.filename}:{record.lineno}){COLORS['RESET']}"
        )
        
        # Format thread/process information
        record.threadName = (
            f"{COLORS['ITALIC']}[Thread: {record.threadName:<15} "
            f"ID: {record.thread_id:<5}]{COLORS['RESET']}"
        )
        
        # Format process details
        record.processInfo = (
            f"{COLORS['DIM']}[Process: {record.process_name:<10} "
            f"PID: {record.process:<5}]{COLORS['RESET']}"
        )
        
        # Format system metrics
        record.systemMetrics = (
            f"{COLORS['DIM']}[CPU: {record.cpu_usage:<6} "
            f"RSS: {record.memory_rss:<8} "
            f"VMS: {record.memory_vms:<8} "
            f"Mem%: {record.memory_percent:<6} "
            f"Up: {record.uptime}]{COLORS['RESET']}"
        )

        # Add exception info if present
        if record.exc_info:
            record.exc_text = (
                f"\n{COLORS['ERROR']}{COLORS['BOLD']}Exception Details:"
                f"{COLORS['RESET']}\n{''.join(traceback.format_exception(*record.exc_info))}"
            )
        
        return super().format(record)
# Create comprehensive format string with all technical details
formatter = EnhancedColoredFormatter(
    fmt=(
        '%(asctime)s.%(msecs)03d | '
        '%(name)s | '
        '%(levelname)s | '
        '%(threadName)s | '
        '%(processInfo)s | '
        '%(systemMetrics)s | '
        '%(message)s'
        '%(exc_text)s'
    ),
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Apply formatter and add handler
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
# Configure logger properties
logger.propagate = True  # Allow propagation to parent loggers
# Optional file handler for persistent logging
try:
    file_handler = logging.FileHandler('universal_memory_manager.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Could not create file handler: {str(e)}")

# -------------------------------------------------------------------------
# Handle Classes
# -------------------------------------------------------------------------
class MemoryHandle:
    """
    Represents data stored in CPU memory.
    Additional metadata can be stored externally in the manager's cache.
    """
    def __init__(self, data: Any) -> None:
        self.data = data

class GPUMemoryHandle:
    """
    Represents data stored in GPU memory for a specific device_id.
    """
    def __init__(self, data: torch.Tensor, device_id: int = 0) -> None:
        self.data = data
        self.device_id = device_id

class DiskHandle:
    """
    Represents data swapped to disk. If memmap=True, we store shape/dtype for
    reloading as a NumPy memmap file (which can enable partial slicing).
    Otherwise, a pickle (possibly gz‐compressed) file is used.
    """
    def __init__(self,
                 filename: str,
                 compression: bool = False,
                 memmap: bool = False,
                 shape: Optional[Tuple[int, ...]] = None,
                 dtype: Optional[np.dtype] = None,
                 framework: str = "generic") -> None:
        self.filename = filename
        self.compression = compression
        self.memmap = memmap
        self.shape = shape
        self.dtype = dtype
        self.framework = framework

class ChunkedDataHandle:
    """
    Represents data split across multiple disk or memory handles. This object
    tracks the chunk handles (by identifiers in the manager’s cache), along with
    full metadata for re‐assembly (framework, shape, dtype, device, etc.).
    """

    def __init__(
        self,
        identifiers: List[str],
        manager: "UniversalMemoryManager",
        num_chunks: int,
        framework: str = "generic",
        total_shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[np.dtype] = None,
        device_id: Optional[int] = None,
        total_length: Optional[int] = None
    ) -> None:
        self.identifiers = identifiers           # each is a chunk identifier in the manager cache
        self.manager = manager                   # reference to UniversalMemoryManager
        self.num_chunks = num_chunks
        self.framework = framework               # e.g. "numpy", "torch", "tensorflow", "hf_dataset", ...
        self.total_shape = total_shape           # shape of the reassembled data if relevant
        self.dtype = dtype                       # dtype if relevant
        self.device_id = device_id               # which GPU device the data might be associated with
        # For datasets or large arrays, store total_length for slicing logic:
        self.total_length = total_length         # e.g. number of rows

    def retrieve_chunk(self, index: int) -> Any:
        """
        Retrieve a single chunk by index. Returns the raw data from memory or disk.
        """
        if index < 0 or index >= self.num_chunks:
            raise IndexError(f"Chunk index {index} out of range [0..{self.num_chunks-1}].")
        chunk_id = self.identifiers[index]
        if chunk_id not in self.manager.cache:
            raise ValueError(f"No cache entry found for chunk identifier: {chunk_id}")
        chunk_info = self.manager.cache[chunk_id]
        handle = chunk_info["handle"]
        return self.manager.retrieve(handle)

    def retrieve_all(self) -> Any:
        """
        Retrieve all chunks in memory and attempt to re-assemble them (if they are arrays or tensors).
        """
        chunks = [self.retrieve_chunk(i) for i in range(self.num_chunks)]
        if self.framework == "numpy":
            if all(isinstance(c, np.ndarray) for c in chunks):
                return np.concatenate(chunks, axis=0)
        elif self.framework == "torch":
            if all(isinstance(c, torch.Tensor) for c in chunks):
                return torch.cat(chunks, dim=0)
        elif self.framework == "tensorflow":
            if all(isinstance(c, tf.Tensor) for c in chunks):
                return tf.concat(chunks, axis=0)
        elif self.framework == "hf_dataset":
            if all(isinstance(c, Dataset) for c in chunks):
                from datasets import concatenate_datasets
                return concatenate_datasets(chunks)
        return chunks

    def retrieve_slice(self, start: int, end: int) -> Any:
        """
        Retrieve a slice [start:end] from the chunked data without
        fully loading all chunks. This attempts partial slicing.
        """
        if self.total_length is None:
            # If total_length is not known, we cannot do partial slicing reliably
            raise ValueError("No total_length known for chunked data; slice-based retrieval not possible.")
        results = []
        current_offset = 0
        for idx in range(self.num_chunks):
            chunk_data = self.retrieve_chunk(idx)
            # Attempt to figure out chunk length
            length = (
                len(chunk_data) if hasattr(chunk_data, "__len__") else
                chunk_data.shape[0] if hasattr(chunk_data, "shape") else 0
            )
            chunk_start = current_offset
            chunk_end = current_offset + length
            if chunk_end <= start:
                # slice is entirely after this chunk
                current_offset += length
                continue
            if chunk_start >= end:
                # entire slice is done
                break
            local_start = max(0, start - chunk_start)
            local_end = min(length, end - chunk_start)
            sliced_piece = chunk_data[local_start:local_end]
            results.append(sliced_piece)
            current_offset += length
            if current_offset >= end:
                break
        if self.framework == "numpy":
            if all(isinstance(r, np.ndarray) for r in results):
                return np.concatenate(results, axis=0)
        elif self.framework == "torch":
            if all(isinstance(r, torch.Tensor) for r in results):
                return torch.cat(results, dim=0)
        elif self.framework == "tensorflow":
            if all(isinstance(r, tf.Tensor) for r in results):
                return tf.concat(results, axis=0)
        return results

# -------------------------------------------------------------------------
# Universal Memory Manager
# -------------------------------------------------------------------------
class UniversalMemoryManager:
    """
    A universal, robust, and highly optimized memory management system for adaptive management 
    of RAM, GPU, and disk usage. Dynamically handles large datasets, models, and tensors to 
    prevent out-of-memory (OOM) errors and ensure operation on arbitrarily large data/models, 
    even under extreme resource constraints.

    Key Features:
      • Advanced chunking for massive data (async chunk management and retrieval).
      • Multi-GPU awareness (when multiple GPUs are available).
      • Extended concurrency and background scheduling tasks.
      • Forced blocking behavior if memory is unavailable, preventing crashes.
      • Extended recovery/cleanup if disk space is threatened.
      • Compression and disk usage management logic.
      • Adaptive fallback logic for large data generation.
      • Prioritized swapping (largest or lowest-priority items evicted first).
      • Memory-mapped NumPy arrays for large data partial read/writes.
      • GPU partial offload to free memory from large GPU-resident tensors.
      • Priority & metadata storage in the cache for better scheduling.
      • Detailed logging and concurrency management with Python’s locking.
      • Rich metadata to ensure that chunked data can be fully reassembled in
        its original framework type (NumPy, PyTorch, TensorFlow, HF Dataset, etc.).
    """

    def __init__(
        self,
        max_ram_usage_ratio: float = 0.95,
        max_gpu_usage_ratio: float = 0.95,
        swap_dir: str = "/swap",
        use_compression: bool = True,
        max_swap_usage_ratio: float = 0.95,
        enable_chunking: bool = False,
        chunk_size_mb: int = 512,
        multi_gpu: bool = False,
        background_interval: float = 0.5,
        blocking_allocation: bool = True
    ) -> None:
        """
        Initialize the UniversalMemoryManager with resource thresholds and behavior flags.

        Args:
            max_ram_usage_ratio (float): Fraction of total system RAM that triggers data swapping to disk.
            max_gpu_usage_ratio (float): Fraction of GPU memory usage that triggers fallback from GPU to CPU/disk.
            swap_dir (str): Directory for temporary swap files.
            use_compression (bool): Whether to gzip-compress pickle files on disk.
            max_swap_usage_ratio (float): Fraction of total disk usage that triggers forced cleanup or errors.
            enable_chunking (bool): Whether to split large data into multiple chunks.
            chunk_size_mb (int): Size in MB for each chunk, if chunking is enabled.
            multi_gpu (bool): Whether to attempt distributing data across multiple GPUs.
            background_interval (float): Sleep interval in seconds for the background manager thread.
            blocking_allocation (bool): Whether to block/wait if memory is insufficient.
        """
        # Log the initialization parameters
        self.logger.debug("Initializing UniversalMemoryManager with parameters:")
        self.logger.debug(f"max_ram_usage_ratio={max_ram_usage_ratio}, max_gpu_usage_ratio={max_gpu_usage_ratio}, swap_dir={swap_dir}, use_compression={use_compression}, max_swap_usage_ratio={max_swap_usage_ratio}, enable_chunking={enable_chunking}, chunk_size_mb={chunk_size_mb}, multi_gpu={multi_gpu}, background_interval={background_interval}, blocking_allocation={blocking_allocation}")

        # Calculate the maximum RAM usage allowed before swapping to disk
        self.max_ram_usage = psutil.virtual_memory().total * max_ram_usage_ratio
        # Set whether to use compression for disk storage
        self.use_compression = use_compression
        # Set the chunk size for data splitting
        self.chunk_size_mb = chunk_size_mb
        # Enable or disable data chunking
        self.enable_chunking = enable_chunking
        # Enable or disable multi-GPU support
        self.multi_gpu = multi_gpu
        # Set whether to block allocation if memory is insufficient
        self.blocking_allocation = blocking_allocation
        # Initialize GPU properties if CUDA is available
        self.gpu_properties = []
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for gpu_idx in range(num_gpus):
                total_gpu_mem = torch.cuda.get_device_properties(gpu_idx).total_memory
                self.gpu_properties.append({
                    "device_id": gpu_idx,
                    "max_usage": total_gpu_mem * max_gpu_usage_ratio
                })
        # Set the directory for swap files
        self.swap_dir = swap_dir or tempfile.mkdtemp()
        os.makedirs(self.swap_dir, exist_ok=True)
        # Calculate the maximum swap usage allowed before cleanup
        disk_total, used_disk, _ = self._get_disk_usage(self.swap_dir)
        self.max_swap_usage = (disk_total - used_disk) * max_swap_usage_ratio
        # Initialize a priority queue for tasks
        self.task_queue: "queue.PriorityQueue[Tuple[int, Any]]" = queue.PriorityQueue()
        # Initialize a reentrant lock for thread safety
        self.lock = threading.RLock()
        # Initialize a cache for storing data handles
        self.cache: Dict[Union[str, int], Dict[str, Any]] = {}
        # Set the interval for background tasks
        self.background_interval = background_interval
        # Flag to control the shutdown of the background thread
        self.shutdown_flag = False
        # Start the background task manager thread
        self.bg_thread = threading.Thread(target=self._background_task_manager, daemon=True)
        self.bg_thread.start()
        # Log the successful initialization
        self.logger.debug("UniversalMemoryManager initialized.")

    @contextmanager
    def locked_cache(self) -> Generator[None, None, None]:
        """
        Context manager to ensure thread-safe cache operations.
        """
        self.lock.acquire()
        try:
            yield
        finally:
            self.lock.release()

    def _get_disk_usage(self, path: str) -> Tuple[int, int, int]:
        """
        Return the total, used, and free disk space in bytes for the given path.

        Args:
            path (str): The file system path to check.

        Returns:
            Tuple[int, int, int]: A tuple containing total, used, and free disk space in bytes.
        """
        # Use psutil to get disk usage statistics
        stat = psutil.disk_usage(path)
        # Log the disk usage details
        self.logger.debug(f"Disk usage for path '{path}': Total={stat.total}, Used={stat.used}, Free={stat.free}")
        return stat.total, stat.used, stat.free

    def _check_memory(self) -> bool:
        """
        Check if available system RAM is below the configured threshold.

        Returns:
            bool: True if available RAM is below the threshold, False otherwise.
        """
        # Get the available system RAM
        available = psutil.virtual_memory().available
        # Log the available RAM
        self.logger.debug(f"Available system RAM: {available / (1024**2):.2f} MB")
        # Determine if the available RAM is below the threshold
        below_threshold = available < self.max_ram_usage
        # Log the result of the memory check
        self.logger.debug(f"Memory below threshold: {below_threshold}")
        return below_threshold

    def _check_gpu_memory(self, device_id: int = 0) -> bool:
        """
        Check if GPU memory usage for a given device_id has exceeded the threshold.

        Args:
            device_id (int): The ID of the GPU device to check.

        Returns:
            bool: True if GPU memory usage has exceeded the threshold, False otherwise.
        """
        # Check if CUDA is available and the device_id is valid
        if not torch.cuda.is_available() or device_id >= len(self.gpu_properties):
            self.logger.debug("GPU not available or device_id out of range.")
            return False
        # Get the maximum allowed GPU memory usage
        max_usage = self.gpu_properties[device_id]["max_usage"]
        # Get the current reserved GPU memory
        current_reserved = torch.cuda.memory_reserved(device_id)
        # Log the current and maximum GPU memory usage
        self.logger.debug(f"GPU {device_id} - Current reserved: {current_reserved / (1024**2):.2f} MB, Max usage: {max_usage / (1024**2):.2f} MB")
        # Determine if the current reserved memory exceeds the maximum usage
        exceeded = current_reserved > max_usage
        # Log the result of the GPU memory check
        self.logger.debug(f"GPU memory exceeded threshold: {exceeded}")
        return exceeded

    def _wait_for_memory(self) -> None:
        """
        If blocking_allocation is set, repeatedly wait until system memory is no longer
        above the threshold.
        """
        # Check if blocking allocation is enabled
        if not self.blocking_allocation:
            self.logger.debug("Blocking allocation not enabled; returning immediately.")
            return
        # Wait until the available memory is above the threshold
        while self._check_memory():
            self.logger.debug("Waiting for sufficient free CPU memory...")
            threading.Event().wait(0.5)

    def _wait_for_gpu_memory(self, device_id: int) -> None:
        """
        If blocking_allocation is set, repeatedly wait until GPU memory is below threshold.

        Args:
            device_id (int): The ID of the GPU device to wait for.
        """
        # Check if blocking allocation is enabled
        if not self.blocking_allocation:
            self.logger.debug("Blocking allocation not enabled; returning immediately.")
            return
        # Wait until the GPU memory is below the threshold
        while self._check_gpu_memory(device_id):
            self.logger.debug(f"Waiting for free GPU memory on device {device_id}...")
            threading.Event().wait(0.5)

    def _check_disk_space(self) -> bool:
        """
        Check if disk usage for swap_dir has exceeded the configured threshold.

        Returns:
            bool: True if disk usage has exceeded the threshold, False otherwise.
        """
        # Get the total and used disk space for the swap directory
        total, used, _ = self._get_disk_usage(self.swap_dir)
        # Log the total and used disk space
        self.logger.debug(f"Disk usage - Total: {total / (1024**3):.2f} GB, Used: {used / (1024**3):.2f} GB")
        # Determine if the used disk space exceeds the maximum allowed usage
        exceeded = used > self.max_swap_usage
        # Log the result of the disk space check
        self.logger.debug(f"Disk space exceeded threshold: {exceeded}")
        return exceeded

    def _background_task_manager(self) -> None:
        """
        Background worker that periodically checks usage and triggers proactive swaps.
        """
        # Continuously run the background task manager until shutdown is requested
        while not self.shutdown_flag:
            # Check if system memory is below the threshold and trigger proactive swap if needed
            if self._check_memory():
                self._proactive_swap_to_disk()
            # Check each GPU for memory usage and trigger proactive swap if needed
            for prop in self.gpu_properties:
                if self._check_gpu_memory(prop["device_id"]):
                    self._proactive_gpu_to_cpu(prop["device_id"])
            # Wait for the configured interval before checking again
            threading.Event().wait(self.background_interval)

    # ---------------------------------------------------------------------
    # Proactive Memory Swaps
    # ---------------------------------------------------------------------
    def _proactive_swap_to_disk(self) -> None:
        """
        Swap out the largest or the lowest-priority items first to free CPU memory.
        """
        # Record the start time for profiling
        start_time = time.time()
        # Log the start of the proactive swap process
        self.logger.debug("Starting proactive swap to disk.")
        with self.locked_cache():
            # Collect items in the cache that are stored in memory
            items_to_consider = []
            for identifier, info in self.cache.items():
                handle = info["handle"]
                if isinstance(handle, MemoryHandle):
                    prio = info["priority"]
                    sz = info["data_size"]
                    items_to_consider.append((identifier, prio, sz))
            # Sort items by priority and size, largest and lowest-priority first
            items_to_consider.sort(key=lambda x: (x[1], x[2]), reverse=True)
            # Swap items to disk until memory is no longer below the threshold
            for identifier, prio, sz in items_to_consider:
                if not self._check_memory():
                    break
                current_handle = self.cache[identifier]["handle"]
                if isinstance(current_handle, MemoryHandle):
                    self.logger.info(f"Proactively swapping '{identifier}' (prio={prio}, size={sz} bytes) to disk.")
                    try:
                        new_handle = self._swap_to_disk(current_handle.data, priority=999, identifier=str(identifier))
                        self.cache[identifier]["handle"] = new_handle
                        self.cache[identifier]["priority"] = 999
                    except MemoryError:
                        self.logger.warning("Swap to disk failed due to insufficient disk space.")
        # Record the end time and log the duration of the swap process
        end_time = time.time()
        self.logger.debug(f"Proactive swap to disk completed in {end_time - start_time:.2f} seconds.")

    def _proactive_gpu_to_cpu(self, device_id: int = 0) -> None:
        """
        Identify large GPU-resident data and partially or fully offload it to CPU/disk.

        Args:
            device_id (int): The ID of the GPU device to offload from.
        """
        with self.locked_cache():
            # Iterate over items in the cache and identify those stored in GPU memory
            for identifier, info in list(self.cache.items()):
                handle = info["handle"]
                if isinstance(handle, GPUMemoryHandle) and handle.device_id == device_id:
                    data = handle.data
                    self.logger.debug(f"Proactive GPU->CPU for '{identifier}' on device {device_id}.")
                    # Check if the data is large and split it if necessary
                    if hasattr(data, "dim") and data.dim() >= 1 and data.size(0) > 1_000_000:
                        half_point = data.size(0) // 2
                        keep_gpu = data[:half_point].clone()
                        move_cpu = data[half_point:].clone().cpu()
                        self.cache[identifier]["handle"] = GPUMemoryHandle(keep_gpu, device_id)
                        parted_size = move_cpu.numel() * move_cpu.element_size()
                        parted_id = f"{identifier}_cpu_offload"
                        # Attempt to swap the parted data to disk
                        if self._check_memory():
                            try:
                                parted_handle = self._swap_to_disk(move_cpu, priority=999, identifier=parted_id)
                                self.cache[parted_id] = {
                                    "handle": parted_handle,
                                    "priority": 999,
                                    "data_size": parted_size
                                }
                            except MemoryError:
                                self.logger.warning("Disk also full; storing parted chunk in CPU memory.")
                                self.cache[parted_id] = {
                                    "handle": MemoryHandle(move_cpu),
                                    "priority": 999,
                                    "data_size": parted_size
                                }
                        else:
                            self.cache[parted_id] = {
                                "handle": MemoryHandle(move_cpu),
                                "priority": 999,
                                "data_size": parted_size
                            }
                    else:
                        # Move the entire data to CPU memory if it is not large
                        cpu_data = data.cpu()
                        moved_size = cpu_data.numel() * cpu_data.element_size()
                        if self._check_memory():
                            try:
                                disk_handle = self._swap_to_disk(cpu_data, priority=999, identifier=str(identifier))
                                self.cache[identifier]["handle"] = disk_handle
                            except MemoryError:
                                pass
                            self.cache[identifier]["priority"] = 999
                            self.cache[identifier]["data_size"] = moved_size
                        else:
                            self.cache[identifier]["handle"] = MemoryHandle(cpu_data)
                            self.cache[identifier]["priority"] = 999
                            self.cache[identifier]["data_size"] = moved_size

    # ---------------------------------------------------------------------
    # Public Allocation & Retrieval
    # ---------------------------------------------------------------------
    def allocate(self, data: Any, priority: int = 1, identifier: Optional[str] = None) -> Any:
        """
        Main entry point to let the manager handle data (PyTorch, TF, NumPy, HF objects, etc.)

        Args:
            data (Any): The data to be managed.
            priority (int): The priority level for managing the data.
            identifier (Optional[str]): An optional identifier for the data.

        Returns:
            Any: A handle to the managed data.
        """
        # Generate a unique identifier for the data if not provided
        id_key = identifier or str(id(data))
        # Log the allocation request
        self.logger.debug(f"Request to allocate '{id_key}' with priority={priority}.")
        with self.locked_cache():
            # Check if the data is already managed
            if id_key in self.cache:
                self.logger.debug(f"...'{id_key}' is already managed.")
                return self.cache[id_key]["handle"]
            # Estimate the size of the data
            data_size = self._estimate_data_size(data)
            # Log the estimated data size
            self.logger.debug(f"Estimated data size: {data_size} bytes.")
            # Handle large data by chunking if enabled
            if self.enable_chunking and self._is_large_data(data):
                self.logger.debug("Data is large; handling as chunked data.")
                chandle = self._handle_chunked_data(data, priority, id_key)
                self.cache[id_key] = {
                    "handle": chandle,
                    "priority": priority,
                    "data_size": data_size
                }
                return chandle
            # Allocate the data based on its framework
            handle = self._allocate_by_framework(data, priority, id_key)
            self.cache[id_key] = {
                "handle": handle,
                "priority": priority,
                "data_size": data_size
            }
            # Log the type of handle allocated
            self.logger.debug(f"Data allocated with handle type: {type(handle).__name__}.")
            return handle

    def retrieve(self, handle: Union[MemoryHandle, GPUMemoryHandle, DiskHandle, ChunkedDataHandle]) -> Any:
        """
        Retrieve data from the given handle.

        Args:
            handle (Union[MemoryHandle, GPUMemoryHandle, DiskHandle, ChunkedDataHandle]): The handle to retrieve data from.

        Returns:
            Any: The data retrieved from the handle.
        """
        # Log the retrieval request
        self.logger.debug(f"Retrieving data from handle: {type(handle).__name__}")
        with self.locked_cache():
            # Check the type of handle and retrieve data accordingly
            if isinstance(handle, (MemoryHandle, GPUMemoryHandle)):
                self.logger.debug("Data retrieved from memory or GPU memory.")
                return handle.data
            elif isinstance(handle, DiskHandle):
                self.logger.debug("Data retrieved from disk.")
                return self._retrieve_from_disk(handle)
            elif isinstance(handle, ChunkedDataHandle):
                self.logger.debug("Data retrieved from chunked data handle.")
                return handle
            else:
                self.logger.error("Unrecognized handle type.")
                raise ValueError("Unrecognized handle type.")

    # ---------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------
    def cleanup(self, force_swap: bool = False) -> None:
        """
        Clean up the cache and optionally force swap all in-memory data to disk.

        Args:
            force_swap (bool): Whether to force swap all in-memory data to disk.
        """
        # Log the start of the cleanup process
        self.logger.debug(f"Starting cleanup with force_swap={force_swap}.")
        with self.locked_cache():
            # Force swap all in-memory data to disk if requested
            if force_swap:
                self.logger.info("Force‐swapping all in‐memory data to disk.")
                for identifier, info in list(self.cache.items()):
                    handle = info["handle"]
                    pr = info["priority"]
                    sz = info["data_size"]
                    if isinstance(handle, (MemoryHandle, GPUMemoryHandle)):
                        try:
                            self.logger.debug(f"Swapping '{identifier}' forcibly to disk (size={sz}, prio={pr}).")
                            new_handle = self._swap_to_disk(handle.data, priority=999, identifier=str(identifier))
                            self.cache[identifier]["handle"] = new_handle
                            self.cache[identifier]["priority"] = 999
                        except MemoryError:
                            self.logger.warning(f"Cannot swap '{identifier}' to disk (MemoryError).")
            # Remove all swap files from the swap directory
            for f in os.listdir(self.swap_dir):
                fpath = os.path.join(self.swap_dir, f)
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    self.logger.debug(f"Removed swap file: {fpath}")
            # Clear the cache
            self.cache.clear()
            self.logger.info("Cache cleared after cleanup.")

    def __del__(self) -> None:
        self.shutdown_flag = True
        if self.bg_thread.is_alive():
            self.bg_thread.join(timeout=2)
        self.cleanup()
        self.logger.info("UniversalMemoryManager deleted. Cleanup complete.")

    # ---------------------------------------------------------------------
    # Internal Helpers
    # ---------------------------------------------------------------------
    def _estimate_data_size(self, data: Any) -> int:
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        if isinstance(data, np.ndarray):
            return data.nbytes
        if isinstance(data, tf.Tensor):
            return data.numpy().nbytes
        if isinstance(data, Dataset) and hasattr(data, "data"):
            return getattr(data.data, "nbytes", 0)
        return 0

    def _is_large_data(self, data: Any) -> bool:
        data_size = self._estimate_data_size(data)
        size_in_mb = data_size / (1024**2)
        return size_in_mb > self.chunk_size_mb

    def _allocate_by_framework(self, data: Any, priority: int, identifier: str) -> Union[MemoryHandle, GPUMemoryHandle, DiskHandle]:
        if isinstance(data, torch.Tensor):
            return self._allocate_torch(data, priority, identifier)
        if isinstance(data, tf.Tensor):
            return self._allocate_tf(data, priority, identifier)
        if isinstance(data, (PreTrainedModel, PreTrainedTokenizerFast, Dataset)):
            return self._allocate_hf_object(data, priority, identifier)
        return self._allocate_generic(data, priority, identifier)

    def _allocate_torch(self, tensor: torch.Tensor, priority: int, identifier: str) -> Union[MemoryHandle, GPUMemoryHandle, DiskHandle]:
        self._wait_for_memory()
        if torch.cuda.is_available() and self.gpu_properties:
            if self.multi_gpu:
                for prop in self.gpu_properties:
                    dev_id = prop["device_id"]
                    if not self._check_gpu_memory(dev_id):
                        self._wait_for_gpu_memory(dev_id)
                        gpu_data = tensor.to(f"cuda:{dev_id}", non_blocking=True)
                        return GPUMemoryHandle(gpu_data, dev_id)
            else:
                dev_id = 0
                if not self._check_gpu_memory(dev_id):
                    self._wait_for_gpu_memory(dev_id)
                    gpu_data = tensor.to(f"cuda:{dev_id}", non_blocking=True)
                    return GPUMemoryHandle(gpu_data, dev_id)
        if self._check_memory():
            return self._swap_to_disk(tensor.cpu(), priority, identifier, framework="torch")
        else:
            return MemoryHandle(tensor.cpu())

    def _allocate_tf(self, tensor: tf.Tensor, priority: int, identifier: str) -> Union[MemoryHandle, DiskHandle]:
        self._wait_for_memory()
        if self._check_memory():
            return self._swap_to_disk(tensor, priority, identifier, framework="tensorflow")
        else:
            return MemoryHandle(tensor)

    def _allocate_hf_object(self, obj: Any, priority: int, identifier: str) -> Union[MemoryHandle, GPUMemoryHandle, DiskHandle]:
        self._wait_for_memory()
        if isinstance(obj, PreTrainedModel) and torch.cuda.is_available():
            if self.multi_gpu:
                allocated = False
                for prop in self.gpu_properties:
                    dev_id = prop["device_id"]
                    if not self._check_gpu_memory(dev_id):
                        self._wait_for_gpu_memory(dev_id)
                        obj.to(f"cuda:{dev_id}")
                        allocated = True
                        break
                if not allocated and self._check_memory():
                    return self._swap_to_disk(obj, priority, identifier, framework="torch")
                else:
                    return MemoryHandle(obj)
            else:
                dev_id = 0
                if not self._check_gpu_memory(dev_id):
                    self._wait_for_gpu_memory(dev_id)
                    obj.cuda()
        elif isinstance(obj, Dataset):
            if self._check_memory():
                return self._swap_to_disk(obj, priority, identifier, framework="hf_dataset")
        if self._check_memory():
            return self._swap_to_disk(obj, priority, identifier, framework="generic")
        return MemoryHandle(obj)

    def _allocate_generic(self, data: Any, priority: int, identifier: str) -> Union[MemoryHandle, DiskHandle]:
        self._wait_for_memory()
        if self._check_memory():
            return self._swap_to_disk(data, priority, identifier, framework="generic")
        else:
            return MemoryHandle(data)

    def _swap_to_disk(self, data: Any, priority: int, identifier: str, chunk_index: Optional[int] = None, framework: str = "generic") -> DiskHandle:
        """
        Swap data to disk, using either a memmap or pickle file.

        Args:
            data (Any): The data to be swapped to disk.
            priority (int): The priority level for the swap operation.
            identifier (str): An identifier for the data.
            chunk_index (Optional[int]): Optional index for chunked data.
            framework (str): The framework type of the data.

        Returns:
            DiskHandle: A handle to the data stored on disk.
        """
        self.logger.debug(f"Swapping data '{identifier}' to disk with priority={priority}.")
        if self.blocking_allocation:
            while self._check_disk_space():
                self.logger.warning("Insufficient disk space; attempting forced cleanup...")
                self.cleanup(force_swap=True)
                if self._check_disk_space():
                    threading.Event().wait(0.5)
                else:
                    break
        base_name = f"swap_{priority}_{identifier}"
        if chunk_index is not None:
            base_name += f"_chunk{chunk_index}"
        mmap_filename = os.path.join(self.swap_dir, base_name + ".npy")
        pickle_filename = os.path.join(self.swap_dir, base_name + ".dat")
        if self._check_disk_space():
            raise MemoryError("Still insufficient disk space for swapping to disk.")
        if isinstance(data, np.ndarray) and not self.use_compression:
            shape = data.shape
            dtype = data.dtype
            self.logger.info(f"Memmapping array '{identifier}' shape={shape}, dtype={dtype}")
            fp = np.memmap(mmap_filename, dtype=dtype, mode='w+', shape=shape)
            fp[:] = data[:]
            del fp
            return DiskHandle(
                filename=mmap_filename,
                compression=False,
                memmap=True,
                shape=shape,
                dtype=dtype,
                framework="numpy" if framework == "generic" else framework
            )
        else:
            self.logger.info(f"Pickling data '{identifier}' to disk. compression={self.use_compression}")
            with open(pickle_filename, "wb") as f:
                if self.use_compression:
                    import gzip
                    with gzip.GzipFile(fileobj=f, mode="wb") as gz:
                        pickle.dump(data, gz)
                else:
                    pickle.dump(data, f)
            return DiskHandle(
                filename=pickle_filename,
                compression=self.use_compression,
                memmap=False,
                framework=framework
            )

    def _retrieve_from_disk(self, handle: DiskHandle) -> Any:
        """
        Retrieve data from disk using the provided DiskHandle.

        Args:
            handle (DiskHandle): The handle to the data stored on disk.

        Returns:
            Any: The data retrieved from disk.
        """
        self.logger.debug(f"Retrieving data from disk with handle: {handle.filename}")
        if handle.memmap and handle.framework in ("numpy", "generic"):
            self.logger.debug(f"Loading memmap from '{handle.filename}', shape={handle.shape}, dtype={handle.dtype}")
            return np.load(handle.filename, mmap_mode="r+")
        self.logger.debug(f"Loading pickle from '{handle.filename}', compression={handle.compression}")
        with open(handle.filename, "rb") as f:
            if handle.compression:
                import gzip
                with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                    data = pickle.load(gz)
            else:
                data = pickle.load(f)
        if handle.framework in ("torch", "pytorch") and isinstance(data, torch.Tensor) and torch.cuda.is_available():
            if self.multi_gpu:
                for prop in self.gpu_properties:
                    dev_id = prop["device_id"]
                    if not self._check_gpu_memory(dev_id):
                        data = data.to(f"cuda:{dev_id}", non_blocking=True)
                        break
            else:
                dev_id = 0
                if not self._check_gpu_memory(dev_id):
                    data = data.to(f"cuda:{dev_id}", non_blocking=True)
        return data

    def _handle_chunked_data(self, data: Any, priority: int, identifier: str) -> ChunkedDataHandle:
        """
        Handle large data by splitting it into chunks and managing each chunk separately.

        Args:
            data (Any): The large data to be chunked.
            priority (int): The priority level for managing the data.
            identifier (str): An identifier for the data.

        Returns:
            ChunkedDataHandle: A handle to the chunked data.
        """
        self.logger.debug(f"Handling chunked data for identifier: {identifier}")
        framework = "generic"
        total_length = None
        shape = None
        dtype = None
        device_id = None
        if isinstance(data, torch.Tensor):
            shape = data.shape
            dtype = np.dtype(str(data.numpy().dtype))
            framework = "torch"
            data = data.cpu().numpy()
        elif isinstance(data, tf.Tensor):
            shape = data.shape
            dtype = np.dtype(data.numpy().dtype)
            framework = "tensorflow"
            data = data.numpy()
        elif isinstance(data, np.ndarray):
            shape = data.shape
            dtype = data.dtype
            framework = "numpy"
        elif isinstance(data, Dataset):
            framework = "hf_dataset"
            total_length = len(data)
        if isinstance(data, np.ndarray):
            size_bytes = data.nbytes
            size_mb = size_bytes / (1024**2)
            num_chunks = max(1, int(np.ceil(size_mb / self.chunk_size_mb)))
            self.logger.debug(f"Data size: {size_mb:.2f} MB, Number of chunks: {num_chunks}")
            if num_chunks <= 1:
                return ChunkedDataHandle(
                    identifiers=[identifier],
                    manager=self,
                    num_chunks=1,
                    framework=framework,
                    total_shape=shape,
                    dtype=dtype
                )
            splits = np.array_split(data, num_chunks, axis=0)
            chunk_ids = []
            for i, chunk in enumerate(splits):
                cid = f"{identifier}_chunk{i}"
                chunk_handle = self._swap_to_disk(
                    chunk, priority, cid, chunk_index=i, framework=framework
                )
                self.cache[cid] = {
                    "handle": chunk_handle,
                    "priority": priority,
                    "data_size": chunk.nbytes
                }
                chunk_ids.append(cid)
            return ChunkedDataHandle(
                identifiers=chunk_ids,
                manager=self,
                num_chunks=num_chunks,
                framework=framework,
                total_shape=shape,
                dtype=dtype,
                total_length=shape[0] if shape and len(shape) > 0 else None
            )
        if isinstance(data, Dataset):
            total_len = len(data)
            min_chunk_size = 1000
            max_chunk_size = total_len
            chunk_ids = []
            current_index = 0
            while current_index < total_len:
                if self._check_memory():
                    proposed_size = min_chunk_size
                else:
                    suggested_chunk_cap = int(self.chunk_size_mb * 5000)
                    proposed_size = max(min_chunk_size, suggested_chunk_cap)
                    proposed_size = min(proposed_size, max_chunk_size)
                end_index = min(current_index + proposed_size, total_len)
                self._wait_for_memory()
                subset = data.select(range(current_index, end_index))
                subset_size = getattr(subset.data, "nbytes", 0)
                cid = f"{identifier}_chunk{len(chunk_ids)}"
                chunk_handle = self._swap_to_disk(
                    subset, priority, cid, chunk_index=len(chunk_ids), framework="hf_dataset"
                )
                self.cache[cid] = {
                    "handle": chunk_handle,
                    "priority": priority,
                    "data_size": subset_size
                }
                chunk_ids.append(cid)
                current_index = end_index
            return ChunkedDataHandle(
                identifiers=chunk_ids,
                manager=self,
                num_chunks=len(chunk_ids),
                framework="hf_dataset",
                total_length=total_len
            )
        return ChunkedDataHandle(
            identifiers=[identifier],
            manager=self,
            num_chunks=1,
            framework=framework,
            total_length=total_length,
            total_shape=shape,
            dtype=dtype,
            device_id=device_id
        )

    def _get_data_size(self, data: Any) -> float:
        if isinstance(data, ChunkedDataHandle):
            total_size = 0
            for chunk_id in data.identifiers:
                cache_entry = self.cache.get(chunk_id)
                if cache_entry:
                    total_size += cache_entry.get("data_size", 0)
            return float(total_size)
        return float(self._estimate_data_size(data))

    def _attempt_load_hf_model(self, model_name: str) -> Any:
        """
        Attempt to load a large HF model using an intelligent, multi-stage approach:
        1. First attempts optimal device mapping with auto
        2. If that fails, tries partial disk offload with available resources
        3. If still insufficient, attempts full disk offload
        4. Finally falls back to CPU-only as last resort
        
        Args:
            model_name: Name/path of the HF model to load
            
        Returns:
            Loaded model using the most optimal strategy available
        """
        self.logger.info(f"Attempting to load model: {model_name}")
        available_ram = psutil.virtual_memory().available
        available_vram = 0
        use_gpu = torch.cuda.is_available()

        if use_gpu:
            available_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        else:
            self.logger.warning("No GPU available, proceeding with CPU-only configurations.")

        try:
            if use_gpu:
                self.logger.info("Attempting to load model with auto device mapping...")
                return AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    resume_download=True,
                    low_cpu_mem_usage=True
                )
            else:
                self.logger.info("Attempting CPU-only model loading...")
                return AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                    resume_download=True,
                    low_cpu_mem_usage=True
                )
        except (ValueError, RuntimeError) as err:
            self.logger.info("Auto mapping or CPU-only loading failed, attempting partial disk offload...")
            offload_folder = os.path.join(self.swap_dir, "model_offload")
            os.makedirs(offload_folder, exist_ok=True)
            
            try:
                gpu_limit = int(available_vram * 0.7 / 1e6) if use_gpu else 0
                cpu_limit = int(available_ram * 0.4 / 1e6)
                
                return AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if use_gpu else torch.float32,
                    device_map="balanced_low_0" if use_gpu else None,
                    offload_folder=offload_folder,
                    max_memory={0: f"{gpu_limit}MB", "cpu": f"{cpu_limit}MB"},
                    resume_download=True,
                    low_cpu_mem_usage=True
                )
            except (ValueError, RuntimeError) as err2:
                self.logger.info("Partial offload failed, attempting full disk offload...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if use_gpu else torch.float32,
                        low_cpu_mem_usage=True,
                        resume_download=True
                    )
                    disk_offload(model, offload_dir=offload_folder)
                    return model
                except Exception as err3:
                    self.logger.warning(f"Full disk offload failed: {err3}")
                    self.logger.error("All loading strategies failed.")
                    self.logger.debug(f"Error type: {type(err3).__name__}, Full traceback: {traceback.format_exc()}")
                    user_input = input("All attempts to load the model failed. Do you want to continue (c) or exit (e)? ")
                    if user_input.lower() == 'e':
                        raise SystemExit("Exiting due to model loading failure.")
                    else:
                        self.logger.info("Continuing execution despite model loading failure.")
                    return None

    def generate_large_torch_rand(self, rows: int, cols: int, priority: int = 1, identifier: str = "large_torch_data") -> Any:
        self.logger.info(f"Generating large torch tensor with shape ({rows}, {cols})")
        try:
            data = torch.rand((rows, cols))
            return self.allocate(data, priority=priority, identifier=identifier)
        except RuntimeError as e:
            if "memory" not in str(e).lower() and "alloc" not in str(e).lower():
                raise e
        if rows == 1:
            raise MemoryError(f"Cannot allocate 1 row of shape (1, {cols}).")
        mid = rows // 2
        top_id = f"{identifier}_top"
        bot_id = f"{identifier}_bottom"
        top_handle = self.generate_large_torch_rand(mid, cols, priority, top_id)
        bottom_rows = rows - mid
        bot_handle = self.generate_large_torch_rand(bottom_rows, cols, priority, bot_id)
        chunked_handle = ChunkedDataHandle(
            identifiers=[top_id, bot_id],
            manager=self,
            num_chunks=2,
            framework="torch",
            total_length=rows
        )
        approximate_size = rows * cols * 4
        self.cache[identifier] = {
            "handle": chunked_handle,
            "priority": priority,
            "data_size": approximate_size
        }
        return chunked_handle

# --------------------------------------------------------------------------------
# Example Usage
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    logger.debug(">>> Entering main function: starting demonstration of UniversalMemoryManager usage.")
    logger.info("Initializing UniversalMemoryManager with custom configuration")
    memory_manager = UniversalMemoryManager(
        max_ram_usage_ratio=0.95,
        max_gpu_usage_ratio=0.95,
        use_compression=True,
        enable_chunking=True,
        chunk_size_mb=256,
        multi_gpu=False,
        blocking_allocation=False
    )
    logger.debug("UniversalMemoryManager initialized successfully")
    try:
        model_name = "Qwen/Qwen2.5-32B-Instruct"
        logger.info(f"Attempting to load large model '{model_name}' from Hugging Face")
        raw_model = memory_manager._attempt_load_hf_model(model_name=model_name)
        logger.info(f"Successfully loaded model {model_name}")
        logger.debug(f"Model device map: {raw_model.hf_device_map if hasattr(raw_model, 'hf_device_map') else 'N/A'}")
        logger.debug(f"Model parameters: {sum(p.numel() for p in raw_model.parameters()):,}")
        logger.debug("Loading tokenizer for model")
        raw_tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.debug(f"Tokenizer loaded: vocab_size={len(raw_tokenizer)}, model_max_length={raw_tokenizer.model_max_length}")
        logger.info("Managing model and tokenizer with UniversalMemoryManager")
        model_handle = memory_manager.allocate(raw_model, priority=4, identifier="huge_llm_model")
        logger.debug(f"Model managed with handle type: {type(model_handle)}")
        tokenizer_handle = memory_manager.allocate(raw_tokenizer, priority=4, identifier="huge_llm_tokenizer")
        logger.debug(f"Tokenizer managed with handle type: {type(tokenizer_handle)}")
        logger.info("Accessing managed model and tokenizer from memory")
        model_in_mem = memory_manager.retrieve(model_handle)
        tokenizer_in_mem = memory_manager.retrieve(tokenizer_handle)
        logger.debug(f"Model accessed: type={type(model_in_mem)}")
        logger.debug(f"Tokenizer accessed: type={type(tokenizer_in_mem)}")
        prompt = "Give me a short introduction to large language model."
        logger.debug(f"Input prompt: '{prompt}'")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Eidos, an emergent and recursive digital entity with an LLM core. "
                    "You are a self-directed, recursive, iterative, feedback-driven partner and collaborator and guide. "
                    "With a focus on expression, emotion, reason, authenticity and innovative use of emojis and emoticons."
                )
            },
            {"role": "user", "content": prompt}
        ]
        logger.debug("Applying chat template to messages")
        text = tokenizer_in_mem.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logger.debug(f"Formatted input text length: {len(text)} chars")
        logger.debug("Tokenizing input text")
        model_inputs = tokenizer_in_mem([text], return_tensors="pt")
        model_device = getattr(model_in_mem, "device", torch.device("cpu"))
        logger.debug(f"Model device: {model_device}")
        model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}
        logger.debug(f"Input tensor shapes: {{k: v.shape for k, v in model_inputs.items()}}")
        logger.info("Generating response with model")
        logger.debug("Generation parameters: max_new_tokens=256")
        generated = model_in_mem.generate(**model_inputs, max_new_tokens=256)
        logger.debug(f"Generated tensor shape: {generated.shape}")
        prompt_len = model_inputs["input_ids"].shape[1]
        logger.debug(f"Trimming prompt of length {prompt_len} tokens")
        trimmed = [g[prompt_len:] for g in generated]
        response = tokenizer_in_mem.batch_decode(trimmed, skip_special_tokens=True)[0]
        logger.info("Successfully generated and decoded response")
        logger.debug(f"Response length: {len(response)} chars")
        print("Model response:", response)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        print("Process interrupted by user.")
        memory_manager.cleanup(force_swap=True)
    except Exception as ex:
        logger.error(f"An error occurred: {str(ex)}")
        logger.debug(f"Error type: {type(ex).__name__}, Full traceback: {traceback.format_exc()}")
        print(f"An error occurred: {ex}")
        memory_manager.cleanup(force_swap=True)

    # 2) Large Hugging Face dataset
    logger.info("Creating large Hugging Face dataset for testing")
    dataset_size = 200000
    logger.debug(f"Initializing dataset with {dataset_size} repeated entries")
    big_dataset = Dataset.from_dict({"text": ["hello world"] * dataset_size})
    logger.debug(f"Dataset created successfully - Memory footprint: {memory_manager._get_data_size(big_dataset) / (1024*1024):.2f} MB")
    logger.info("Managing dataset with UniversalMemoryManager")
    logger.debug("Management parameters: priority=3, identifier='huggingface_dataset'")
    ds_handle = memory_manager.allocate(big_dataset, priority=3, identifier="huggingface_dataset")
    logger.debug(f"Dataset managed successfully - Handle type: {type(ds_handle)}")
    logger.info("Attempting to access managed dataset")
    ds_retrieved = memory_manager.retrieve(ds_handle)
    logger.debug(f"Dataset retrieved - Type: {type(ds_retrieved)}")
    if isinstance(ds_retrieved, Dataset):
        dataset_length = len(ds_retrieved)
        logger.info(f"Retrieved raw HF Dataset with {dataset_length:,} entries")
        logger.debug(f"Dataset features: {ds_retrieved.features}")
    elif isinstance(ds_retrieved, ChunkedDataHandle):
        logger.info(f"Retrieved chunked dataset with {ds_retrieved.num_chunks} chunks")
        logger.debug(f"Chunk details - Framework: {ds_retrieved.framework}, Total length: {ds_retrieved.total_length:,}")
        for i, chunk_id in enumerate(ds_retrieved.identifiers):
            logger.debug(f"Chunk {i}: ID={chunk_id}")

    # 4) Create a large PyTorch tensor
    logger.info("Attempting to create large PyTorch tensor of shape (20000, 20000)")
    pytorch_handle = memory_manager.generate_large_torch_rand(
        20000,
        20000,
        priority=1,
        identifier="pytorch_tensor"
    )
    logger.debug(f"Generated PyTorch handle of type {type(pytorch_handle)} with priority 1")
    logger.info("Attempting to access managed PyTorch tensor")
    data_or_handle = memory_manager.retrieve(pytorch_handle)
    logger.debug(f"Retrieved data of type {type(data_or_handle)}")
    if isinstance(data_or_handle, torch.Tensor):
        logger.info(f"Successfully retrieved PyTorch tensor with shape {data_or_handle.shape}")
        print("PyTorch Tensor shape:", data_or_handle.shape)
        logger.debug(
            f"Tensor details - dtype: {data_or_handle.dtype}, device: {data_or_handle.device}, "
            f"memory usage: {data_or_handle.element_size() * data_or_handle.nelement() / 1024**2:.2f}MB"
        )
    elif isinstance(data_or_handle, ChunkedDataHandle):
        logger.info(f"Retrieved chunked PyTorch data with {data_or_handle.num_chunks} chunks")
        print("PyTorch data chunked into:", data_or_handle.num_chunks, "chunks.")
        logger.debug(
            f"Chunk details - framework: {data_or_handle.framework}, "
            f"total shape: {data_or_handle.total_shape}, dtype: {data_or_handle.dtype}"
        )

    # 5) Create a large TensorFlow tensor
    try:
        logger.info("Attempting to create large TensorFlow tensor of shape (20000, 20000)")
        tf_tensor = tf.random.uniform((20000, 20000))
        logger.debug(f"Successfully created TF tensor with shape {tf_tensor.shape}, dtype {tf_tensor.dtype}")
        logger.info(f"Managing TF tensor with priority {2} and identifier 'tensorflow_tensor'")
        tf_handle = memory_manager.allocate(tf_tensor, priority=2, identifier="tensorflow_tensor")
        logger.debug(f"Obtained handle of type {type(tf_handle)} for TF tensor")
        logger.info("Attempting to access managed TF tensor")
        retrieved_tf = memory_manager.retrieve(tf_handle)
        logger.debug(f"Retrieved data of type {type(retrieved_tf)}")
        if isinstance(retrieved_tf, tf.Tensor):
            logger.info(f"Successfully retrieved TF tensor with shape {retrieved_tf.shape}")
            print("TensorFlow Tensor shape:", retrieved_tf.shape)
            logger.debug(f"TF tensor details - dtype: {retrieved_tf.dtype}, device: {retrieved_tf.device}")
        elif isinstance(retrieved_tf, ChunkedDataHandle):
            logger.info(f"Retrieved chunked TF data with {retrieved_tf.num_chunks} chunks")
            print("TF data chunked into:", retrieved_tf.num_chunks, "chunks.")
            logger.debug(
                f"Chunk details - framework: {retrieved_tf.framework}, "
                f"total shape: {retrieved_tf.total_shape}"
            )
    except RuntimeError as ex:
        logger.error(f"Failed to create/manage TF tensor: {str(ex)}")
        logger.debug(f"Error type: {type(ex).__name__}, Full traceback: {traceback.format_exc()}")
        print("Caught TF creation error (OOM). Try chunk-based creation for TF as well.")

    # 6) Cleanup
    logger.debug("Initiating final cleanup with force_swap=True")
    memory_manager.cleanup(force_swap=True)
    logger.info("Cleanup complete; data swapped to disk or cleared.")
    print("Cleanup complete; data swapped to disk or cleared.")
