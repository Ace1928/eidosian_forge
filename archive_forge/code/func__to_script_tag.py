from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum
import os
import uuid
import webbrowser
import cirq_web
def _to_script_tag(bundle_filename: str) -> str:
    """Dumps the contents of a particular bundle file into a script tag.

    Args:
        bundle_filename: the path to the bundle file

    Returns:
        The bundle file as string (readable by browser) wrapped in HTML script tags.
    """
    bundle_file_path = os.path.join(_DIST_PATH, bundle_filename)
    bundle_file = open(bundle_file_path, 'r', encoding='utf-8')
    bundle_file_contents = bundle_file.read()
    bundle_file.close()
    bundle_html = f'<script>{bundle_file_contents}</script>'
    return bundle_html