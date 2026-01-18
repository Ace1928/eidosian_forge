import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
def _font_finder_darwin():
    locations = ['/Library/Fonts', '/Network/Library/Fonts', '/System/Library/Fonts', '/System/Library/Fonts/Supplemental']
    username = os.getenv('USER')
    if username:
        locations.append(f'/Users/{username}/Library/Fonts')
    strange_root = '/System/Library/Assets/com_apple_MobileAsset_Font3'
    if exists(strange_root):
        strange_locations = os.listdir(strange_root)
        for loc in strange_locations:
            locations.append(f'{strange_root}/{loc}/AssetData')
    fonts = {}
    for location in locations:
        if not exists(location):
            continue
        files = os.listdir(location)
        for file in files:
            name, extension = splitext(file)
            if extension in OpenType_extensions:
                _parse_font_entry_darwin(name, join(location, file), fonts)
    return fonts