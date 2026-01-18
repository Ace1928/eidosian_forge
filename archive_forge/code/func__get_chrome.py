import os
import sys
import time
import logging
import warnings
import percy
import requests
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (
from dash.testing.wait import (
from dash.testing.dash_page import DashPageMixin
from dash.testing.errors import DashAppLoadingError, BrowserError, TestingTimeoutError
from dash.testing.consts import SELENIUM_GRID_DEFAULT
def _get_chrome(self):
    options = self._get_wd_options()
    options.set_capability('loggingPrefs', {'browser': 'SEVERE'})
    options.set_capability('goog:loggingPrefs', {'browser': 'SEVERE'})
    if 'DASH_TEST_CHROMEPATH' in os.environ:
        options.binary_location = os.environ['DASH_TEST_CHROMEPATH']
    options.add_experimental_option('prefs', {'download.default_directory': self.download_path, 'download.prompt_for_download': False, 'download.directory_upgrade': True, 'safebrowsing.enabled': False, 'safebrowsing.disable_download_protection': True})
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options.add_argument('--remote-debugging-port=9222')
    chrome = webdriver.Remote(command_executor=self._remote_url, options=options) if self._remote else webdriver.Chrome(options=options)
    if self._headless:
        chrome.command_executor._commands['send_command'] = ('POST', '/session/$sessionId/chromium/send_command')
        params = {'cmd': 'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': self.download_path}}
        res = chrome.execute('send_command', params)
        logger.debug('enabled headless download returns %s', res)
    chrome.set_window_position(0, 0)
    return chrome