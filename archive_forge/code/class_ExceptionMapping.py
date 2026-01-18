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
class ExceptionMapping:
    """
    :Maps each errorcode in ErrorCode object to corresponding exception
    Please refer to https://www.w3.org/TR/webdriver2/#errors for w3c specification
    """
    NO_SUCH_ELEMENT = NoSuchElementException
    NO_SUCH_FRAME = NoSuchFrameException
    NO_SUCH_SHADOW_ROOT = NoSuchShadowRootException
    STALE_ELEMENT_REFERENCE = StaleElementReferenceException
    ELEMENT_NOT_VISIBLE = ElementNotVisibleException
    INVALID_ELEMENT_STATE = InvalidElementStateException
    UNKNOWN_ERROR = WebDriverException
    ELEMENT_IS_NOT_SELECTABLE = ElementNotSelectableException
    JAVASCRIPT_ERROR = JavascriptException
    TIMEOUT = TimeoutException
    NO_SUCH_WINDOW = NoSuchWindowException
    INVALID_COOKIE_DOMAIN = InvalidCookieDomainException
    UNABLE_TO_SET_COOKIE = UnableToSetCookieException
    UNEXPECTED_ALERT_OPEN = UnexpectedAlertPresentException
    NO_ALERT_OPEN = NoAlertPresentException
    SCRIPT_TIMEOUT = TimeoutException
    IME_NOT_AVAILABLE = ImeNotAvailableException
    IME_ENGINE_ACTIVATION_FAILED = ImeActivationFailedException
    INVALID_SELECTOR = InvalidSelectorException
    SESSION_NOT_CREATED = SessionNotCreatedException
    MOVE_TARGET_OUT_OF_BOUNDS = MoveTargetOutOfBoundsException
    INVALID_XPATH_SELECTOR = InvalidSelectorException
    INVALID_XPATH_SELECTOR_RETURN_TYPER = InvalidSelectorException
    ELEMENT_NOT_INTERACTABLE = ElementNotInteractableException
    INSECURE_CERTIFICATE = InsecureCertificateException
    INVALID_ARGUMENT = InvalidArgumentException
    INVALID_COORDINATES = InvalidCoordinatesException
    INVALID_SESSION_ID = InvalidSessionIdException
    NO_SUCH_COOKIE = NoSuchCookieException
    UNABLE_TO_CAPTURE_SCREEN = ScreenshotException
    ELEMENT_CLICK_INTERCEPTED = ElementClickInterceptedException
    UNKNOWN_METHOD = UnknownMethodException