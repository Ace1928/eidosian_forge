import argparse

    Utility method to download all the dataset files
    for `scipy.datasets` module.

    Parameters
    ----------
    path : str, optional
        Directory path to download all the dataset files.
        If None, default to the system cache_dir detected by pooch.
    