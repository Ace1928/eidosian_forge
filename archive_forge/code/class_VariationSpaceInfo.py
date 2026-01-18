import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
class VariationSpaceInfo(object):
    """
    VF info (axes & instances).
    """

    def __init__(self, face, p_ftmmvar):
        """
        Build a VariationSpaceInfo object given face (freetype.Face) and
        p_ftmmvar (pointer to FT_MM_Var).
        """
        ftmv = p_ftmmvar.contents
        axes = []
        for axidx in range(ftmv.num_axis):
            axes.append(VariationAxis(ftmv.axis[axidx]))
        self.axes = tuple(axes)
        inst = []
        for instidx in range(ftmv.num_namedstyles):
            instinfo = ftmv.namedstyle[instidx]
            nid = instinfo.strid
            name = face.get_best_name_string(nid)
            psid = instinfo.psid
            psname = face.get_best_name_string(psid)
            coords = []
            for cidx in range(len(self.axes)):
                coords.append(instinfo.coords[cidx] / 65536.0)
            inst.append(VariationInstance(name, psname, tuple(coords)))
        self.instances = tuple(inst)