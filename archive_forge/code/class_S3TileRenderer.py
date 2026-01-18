from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
class S3TileRenderer(TileRenderer):

    def render(self, da, level):
        try:
            import boto3
        except ImportError:
            raise ImportError('install boto3 to enable rendering to S3')
        try:
            from urlparse import urlparse
        except ImportError:
            from urllib.parse import urlparse
        s3_info = urlparse(self.output_location)
        bucket = s3_info.netloc
        client = boto3.client('s3')
        for img, x, y, z in super(S3TileRenderer, self).render(da, level):
            tile_file_name = '{}.{}'.format(y, self.tile_format.lower())
            key = os.path.join(s3_info.path, str(z), str(x), tile_file_name).lstrip('/')
            output_buf = BytesIO()
            img.save(output_buf, self.tile_format)
            output_buf.seek(0)
            client.put_object(Body=output_buf, Bucket=bucket, Key=key, ACL='public-read')
        return 'https://{}.s3.amazonaws.com/{}'.format(bucket, s3_info.path)