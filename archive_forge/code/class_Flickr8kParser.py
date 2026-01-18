import glob
import os
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
from .vision import VisionDataset
class Flickr8kParser(HTMLParser):
    """Parser for extracting captions from the Flickr8k dataset web page."""

    def __init__(self, root: str) -> None:
        super().__init__()
        self.root = root
        self.annotations: Dict[str, List[str]] = {}
        self.in_table = False
        self.current_tag: Optional[str] = None
        self.current_img: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.current_tag = tag
        if tag == 'table':
            self.in_table = True

    def handle_endtag(self, tag: str) -> None:
        self.current_tag = None
        if tag == 'table':
            self.in_table = False

    def handle_data(self, data: str) -> None:
        if self.in_table:
            if data == 'Image Not Found':
                self.current_img = None
            elif self.current_tag == 'a':
                img_id = data.split('/')[-2]
                img_id = os.path.join(self.root, img_id + '_*.jpg')
                img_id = glob.glob(img_id)[0]
                self.current_img = img_id
                self.annotations[img_id] = []
            elif self.current_tag == 'li' and self.current_img:
                img_id = self.current_img
                self.annotations[img_id].append(data.strip())