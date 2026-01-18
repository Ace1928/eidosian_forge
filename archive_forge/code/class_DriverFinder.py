import logging
from pathlib import Path
from selenium.common.exceptions import NoSuchDriverException
from selenium.webdriver.common.options import BaseOptions
from selenium.webdriver.common.selenium_manager import SeleniumManager
from selenium.webdriver.common.service import Service
class DriverFinder:
    """Utility to find if a given file is present and executable.

    This implementation is still in beta, and may change.
    """

    @staticmethod
    def get_path(service: Service, options: BaseOptions) -> str:
        path = service.path
        try:
            path = SeleniumManager().driver_location(options) if path is None else path
        except Exception as err:
            msg = f'Unable to obtain driver for {options.capabilities['browserName']} using Selenium Manager.'
            raise NoSuchDriverException(msg) from err
        if path is None or not Path(path).is_file():
            raise NoSuchDriverException(f'Unable to locate or obtain driver for {options.capabilities['browserName']}')
        return path