from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class FzPclOptions(object):
    """
    Wrapper class for struct `fz_pcl_options`. Not copyable or assignable.
    PCL output
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_pcl_preset(self, preset):
        """
        Class-aware wrapper for `::fz_pcl_preset()`.
        	Initialize PCL option struct for a given preset.

        	Currently defined presets include:

        		generic	Generic PCL printer
        		ljet4	HP DeskJet
        		dj500	HP DeskJet 500
        		fs600	Kyocera FS-600
        		lj	HP LaserJet, HP LaserJet Plus
        		lj2	HP LaserJet IIp, HP LaserJet IId
        		lj3	HP LaserJet III
        		lj3d	HP LaserJet IIId
        		lj4	HP LaserJet 4
        		lj4pl	HP LaserJet 4 PL
        		lj4d	HP LaserJet 4d
        		lp2563b	HP 2563B line printer
        		oce9050	Oce 9050 Line printer
        """
        return _mupdf.FzPclOptions_fz_pcl_preset(self, preset)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_parse_pcl_options()`.
        		Parse PCL options.

        		Currently defined options and values are as follows:

        			preset=X	Either "generic" or one of the presets as for fz_pcl_preset.
        			spacing=0	No vertical spacing capability
        			spacing=1	PCL 3 spacing (<ESC>*p+<n>Y)
        			spacing=2	PCL 4 spacing (<ESC>*b<n>Y)
        			spacing=3	PCL 5 spacing (<ESC>*b<n>Y and clear seed row)
        			mode2		Disable/Enable mode 2 graphics compression
        			mode3		Disable/Enable mode 3 graphics compression
        			eog_reset	End of graphics (<ESC>*rB) resets all parameters
        			has_duplex	Duplex supported (<ESC>&l<duplex>S)
        			has_papersize	Papersize setting supported (<ESC>&l<sizecode>A)
        			has_copies	Number of copies supported (<ESC>&l<copies>X)
        			is_ljet4pjl	Disable/Enable HP 4PJL model-specific output
        			is_oce9050	Disable/Enable Oce 9050 model-specific output


        |

        *Overload 2:*
         Default constructor, sets `m_internal` to null.

        |

        *Overload 3:*
         Constructor using raw copy of pre-existing `::fz_pcl_options`.
        """
        _mupdf.FzPclOptions_swiginit(self, _mupdf.new_FzPclOptions(*args))
    __swig_destroy__ = _mupdf.delete_FzPclOptions

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzPclOptions_m_internal_value(self)
    m_internal = property(_mupdf.FzPclOptions_m_internal_get, _mupdf.FzPclOptions_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzPclOptions_s_num_instances_get, _mupdf.FzPclOptions_s_num_instances_set)