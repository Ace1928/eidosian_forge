import os
import time
import numpy as np
import PyChromeDevTools
import ipyvolume as ipv
def _screenshot_data(html_filename, timeout_seconds=10, output_widget=None, format='png', width=None, height=None, fig=None, **headless_kwargs):
    chrome = PyChromeDevTools.ChromeInterface(headless_kwargs)
    chrome.Network.enable()
    chrome.Page.enable()
    chrome.Page.navigate(url=html_filename)
    chrome.wait_event('Page.frameStoppedLoading', timeout=60)
    chrome.wait_event('Page.loadEventFired', timeout=60)
    time.sleep(0.5)
    result = chrome.Runtime.evaluate(expression='ipvss()')
    tries = 0
    while tries < 10:
        try:
            url = result['result']['result']['value']
            return url
        except:
            if 'ipvss' in result['result']['result']['description']:
                tries += 1
                time.sleep(0.5)
            else:
                print('error getting result, return value was:', result)
                raise