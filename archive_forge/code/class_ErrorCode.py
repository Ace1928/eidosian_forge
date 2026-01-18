from typing import Any
from typing import Dict
from typing import Type
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.common.exceptions import ElementNotInteractableException
from selenium.common.exceptions import ElementNotSelectableException
from selenium.common.exceptions import ElementNotVisibleException
from selenium.common.exceptions import ImeActivationFailedException
from selenium.common.exceptions import ImeNotAvailableException
from selenium.common.exceptions import InsecureCertificateException
from selenium.common.exceptions import InvalidArgumentException
from selenium.common.exceptions import InvalidCookieDomainException
from selenium.common.exceptions import InvalidCoordinatesException
from selenium.common.exceptions import InvalidElementStateException
from selenium.common.exceptions import InvalidSelectorException
from selenium.common.exceptions import InvalidSessionIdException
from selenium.common.exceptions import JavascriptException
from selenium.common.exceptions import MoveTargetOutOfBoundsException
from selenium.common.exceptions import NoAlertPresentException
from selenium.common.exceptions import NoSuchCookieException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoSuchFrameException
from selenium.common.exceptions import NoSuchShadowRootException
from selenium.common.exceptions import NoSuchWindowException
from selenium.common.exceptions import ScreenshotException
from selenium.common.exceptions import SessionNotCreatedException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import UnableToSetCookieException
from selenium.common.exceptions import UnexpectedAlertPresentException
from selenium.common.exceptions import UnknownMethodException
from selenium.common.exceptions import WebDriverException
class ErrorCode:
    """Error codes defined in the WebDriver wire protocol."""
    SUCCESS = 0
    NO_SUCH_ELEMENT = [7, 'no such element']
    NO_SUCH_FRAME = [8, 'no such frame']
    NO_SUCH_SHADOW_ROOT = ['no such shadow root']
    UNKNOWN_COMMAND = [9, 'unknown command']
    STALE_ELEMENT_REFERENCE = [10, 'stale element reference']
    ELEMENT_NOT_VISIBLE = [11, 'element not visible']
    INVALID_ELEMENT_STATE = [12, 'invalid element state']
    UNKNOWN_ERROR = [13, 'unknown error']
    ELEMENT_IS_NOT_SELECTABLE = [15, 'element not selectable']
    JAVASCRIPT_ERROR = [17, 'javascript error']
    XPATH_LOOKUP_ERROR = [19, 'invalid selector']
    TIMEOUT = [21, 'timeout']
    NO_SUCH_WINDOW = [23, 'no such window']
    INVALID_COOKIE_DOMAIN = [24, 'invalid cookie domain']
    UNABLE_TO_SET_COOKIE = [25, 'unable to set cookie']
    UNEXPECTED_ALERT_OPEN = [26, 'unexpected alert open']
    NO_ALERT_OPEN = [27, 'no such alert']
    SCRIPT_TIMEOUT = [28, 'script timeout']
    INVALID_ELEMENT_COORDINATES = [29, 'invalid element coordinates']
    IME_NOT_AVAILABLE = [30, 'ime not available']
    IME_ENGINE_ACTIVATION_FAILED = [31, 'ime engine activation failed']
    INVALID_SELECTOR = [32, 'invalid selector']
    SESSION_NOT_CREATED = [33, 'session not created']
    MOVE_TARGET_OUT_OF_BOUNDS = [34, 'move target out of bounds']
    INVALID_XPATH_SELECTOR = [51, 'invalid selector']
    INVALID_XPATH_SELECTOR_RETURN_TYPER = [52, 'invalid selector']
    ELEMENT_NOT_INTERACTABLE = [60, 'element not interactable']
    INSECURE_CERTIFICATE = ['insecure certificate']
    INVALID_ARGUMENT = [61, 'invalid argument']
    INVALID_COORDINATES = ['invalid coordinates']
    INVALID_SESSION_ID = ['invalid session id']
    NO_SUCH_COOKIE = [62, 'no such cookie']
    UNABLE_TO_CAPTURE_SCREEN = [63, 'unable to capture screen']
    ELEMENT_CLICK_INTERCEPTED = [64, 'element click intercepted']
    UNKNOWN_METHOD = ['unknown method exception']
    METHOD_NOT_ALLOWED = [405, 'unsupported operation']