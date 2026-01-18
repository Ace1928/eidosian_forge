from itertools import combinations_with_replacement
import itertools
import numpy as np
from skimage import filters, feature
from skimage.util.dtype import img_as_float32
from concurrent.futures import ThreadPoolExecutor
def _singlescale_basic_features_singlechannel(img, sigma, intensity=True, edges=True, texture=True):
    results = ()
    gaussian_filtered = filters.gaussian(img, sigma=sigma, preserve_range=False)
    if intensity:
        results += (gaussian_filtered,)
    if edges:
        results += (filters.sobel(gaussian_filtered),)
    if texture:
        results += (*_texture_filter(gaussian_filtered),)
    return results