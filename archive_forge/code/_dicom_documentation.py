import sys
import os
import struct
import logging
import numpy as np

        Evaluate the series of dicom files. Together they should make up
        a volumetric dataset. This means the files should meet certain
        conditions. Also some additional information has to be calculated,
        such as the distance between the slices. This method sets the
        attributes for "shape", "sampling" and "info".

        This method checks:
          * that there are no missing files
          * that the dimensions of all images match
          * that the pixel spacing of all images match
        