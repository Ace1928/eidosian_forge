import csv
import os
from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Dict, Any
from parlai.core.build_data import download_multiprocess
from parlai.core.params import Opt
from parlai.core.teachers import AbstractImageTeacher
import parlai.utils.typing as PT
def _download_images(self, opt: Opt):
    """
        Download available IGC images.
        """
    urls = []
    ids = []
    for dt in ['test', 'val']:
        df = os.path.join(self.get_data_path(opt), f'IGC_crowd_{dt}.csv')
        with open(df, newline='\n') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            fields = []
            for i, row in enumerate(reader):
                if i == 0:
                    fields = row
                else:
                    data = dict(zip(fields, row))
                    urls.append(data['url'])
                    ids.append(data['id'])
    os.makedirs(self.get_image_path(opt), exist_ok=True)
    image = Image.new('RGB', (100, 100), color=0)
    image.save(os.path.join(self.get_image_path(opt), self.blank_image_id), 'JPEG')
    download_multiprocess(urls, self.get_image_path(opt), dest_filenames=ids)
    for fp in os.listdir(self.get_image_path(opt)):
        img_path = os.path.join(self.get_image_path(opt), fp)
        if os.path.isfile(img_path):
            try:
                Image.open(img_path).convert('RGB')
            except OSError:
                os.remove(img_path)