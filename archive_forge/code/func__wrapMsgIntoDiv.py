import base64
import json
import logging
import re
import uuid
from xml.dom import minidom
from IPython.display import HTML, display
from rdkit import Chem
from rdkit.Chem import Draw
from . import rdMolDraw2D
def _wrapMsgIntoDiv(uuid, msg, quiet):
    return f'<div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output"id="{uuid}">{('' if quiet else msg)}</div>'