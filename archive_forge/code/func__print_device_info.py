from dataclasses import dataclass
import sys
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
import pygame as pg
import pygame.midi
def _print_device_info():
    for i in range(pygame.midi.get_count()):
        r = pygame.midi.get_device_info(i)
        interf, name, input, output, opened = r
        in_out = ''
        if input:
            in_out = '(input)'
        if output:
            in_out = '(output)'
        print('%2i: interface :%s:, name :%s:, opened :%s:  %s' % (i, interf, name, opened, in_out))