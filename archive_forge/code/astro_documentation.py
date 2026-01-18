import numpy as np
import PIL.Image
import pythreejs
import scipy.interpolate
import ipyvolume as ipv
from ipyvolume.datasets import UrlCached
Create a fake galaxy around the points orbit_x/y/z with N_stars around it.