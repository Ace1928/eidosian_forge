from ctypes import POINTER, c_char_p, c_double, c_int, c_long, c_void_p
from django.contrib.gis.gdal.envelope import OGREnvelope
from django.contrib.gis.gdal.libgdal import lgdal
from django.contrib.gis.gdal.prototypes.generation import (

 This module houses the ctypes function prototypes for OGR DataSource
 related data structures. OGR_Dr_*, OGR_DS_*, OGR_L_*, OGR_F_*,
 OGR_Fld_* routines are relevant here.
