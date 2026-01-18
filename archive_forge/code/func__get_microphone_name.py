import datetime
import platform
import subprocess
from typing import Optional, Tuple, Union
import numpy as np
def _get_microphone_name():
    """
    Retrieve the microphone name in Windows .
    """
    command = ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', '']
    try:
        ffmpeg_devices = subprocess.run(command, text=True, stderr=subprocess.PIPE, encoding='utf-8')
        microphone_lines = [line for line in ffmpeg_devices.stderr.splitlines() if '(audio)' in line]
        if microphone_lines:
            microphone_name = microphone_lines[0].split('"')[1]
            print(f'Using microphone: {microphone_name}')
            return f'audio={microphone_name}'
    except FileNotFoundError:
        print('ffmpeg was not found. Please install it or make sure it is in your system PATH.')
    return 'default'