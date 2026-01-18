import unittest
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import logging as log

        Get a list of MRI scans from a given experiment which has multiple
        scans mixed (i.e. MRScans and MRSpectroscopies, aka OtherDicomScans)
        and assert its length matches the list of scans filtered by type
        'xnat:mrScanData'
        