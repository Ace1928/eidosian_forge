import os
import json
import csv
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union, Optional, Tuple, List
import pandas as pd
import logging
import yaml
import pickle
import configparser
import markdown
import openpyxl
import sqlite3
import PyPDF2
import PIL.Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Label, Toplevel
from PIL import Image, ImageTk
import os
import logging
import json
import pandas as pd

        Attempt to read an unknown file type by initially trying to open it as a text file with UTF-8 encoding and returning its content as a string.
        If this fails due to encoding errors or other issues, it will then attempt to read the file as byte data and load it into a pandas DataFrame.
        This method serves as a comprehensive fallback mechanism to handle unknown file types that cannot be identified or processed through standard methods,
        ensuring a robust and universal approach to accessing the content of any file either as a raw text string or structured byte data.

        :return: The content of the unknown file either as a string or a pandas DataFrame containing the byte data of the file.
        :rtype: Union[str, pd.DataFrame]
        