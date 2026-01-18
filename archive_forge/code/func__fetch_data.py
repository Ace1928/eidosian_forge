import os
import pytest
import shutil
from nipype.interfaces.dcm2nii import Dcm2niix
def _fetch_data(datadir, dicoms):
    try:
        'Fetches some test DICOMs using datalad'
        api.install(path=datadir, source=DICOM_DIR)
        data = os.path.join(datadir, dicoms)
        api.get(path=data, dataset=datadir)
    except IncompleteResultsError as exc:
        pytest.skip('Failed to fetch test data: %s' % str(exc))
    return data