import collections
import io
import math
from urllib.parse import urlparse
import warnings
import weakref
from xml.etree import ElementTree
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.img_transform import warp_array
from cartopy.io import LocatedImage, RasterSource
def _warped_located_image(image, source_projection, source_extent, output_projection, output_extent, target_resolution):
    """
    Reproject an Image from one source-projection and extent to another.

    Returns
    -------
    LocatedImage
        A reprojected LocatedImage, the extent of which is >= the requested
        'output_extent'.

    """
    if source_projection == output_projection:
        extent = output_extent
    else:
        img, extent = warp_array(np.asanyarray(image.convert('RGBA'))[::-1], source_proj=source_projection, source_extent=source_extent, target_proj=output_projection, target_res=np.asarray(target_resolution, dtype=int), target_extent=output_extent, mask_extrapolated=True)
        if np.ma.is_masked(img):
            img[:, :, 3] = np.where(np.any(img.mask, axis=2), 0, img[:, :, 3])
            img = img.data
        image = Image.fromarray(img[::-1])
    return LocatedImage(image, extent)