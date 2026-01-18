import os
import platform
from pygame import __file__ as pygame_main_file
def _append_to_datas(file_path):
    res_path = os.path.join(pygame_folder, file_path)
    if os.path.exists(res_path):
        datas.append((res_path, 'pygame'))