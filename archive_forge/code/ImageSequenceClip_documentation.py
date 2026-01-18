import os
import numpy as np
from imageio import imread
from ..VideoClip import VideoClip

    
    A VideoClip made from a series of images.
    

    Parameters
    -----------

    sequence
      Can be one of these:
      - The name of a folder (containing only pictures). The pictures
        will be considered in alphanumerical order.
      - A list of names of image files. In this case you can choose to
        load the pictures in memory pictures 
      - A list of Numpy arrays representing images. In this last case,
        masks are not supported currently.

    fps
      Number of picture frames to read per second. Instead, you can provide
      the duration of each image with durations (see below)

    durations
      List of the duration of each picture.

    with_mask
      Should the alpha layer of PNG images be considered as a mask ?

    ismask
      Will this sequence of pictures be used as an animated mask.

    Notes
    ------

    If your sequence is made of image files, the only image kept in 


    
    