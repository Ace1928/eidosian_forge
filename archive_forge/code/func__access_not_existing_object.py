import threading
import h5py
def _access_not_existing_object(filename):
    """Create a file and access not existing key"""
    with h5py.File(filename, 'w') as newfile:
        try:
            doesnt_exist = newfile['doesnt_exist'].value
        except KeyError:
            pass