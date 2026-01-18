import os
import sys
import stat
import fnmatch
import collections
import errno
def _make_zipfile(base_name, base_dir, verbose=0, dry_run=0, logger=None, owner=None, group=None, root_dir=None):
    """Create a zip file from all the files under 'base_dir'.

    The output zip file will be named 'base_name' + ".zip".  Returns the
    name of the output zip file.
    """
    import zipfile
    zip_filename = base_name + '.zip'
    archive_dir = os.path.dirname(base_name)
    if archive_dir and (not os.path.exists(archive_dir)):
        if logger is not None:
            logger.info('creating %s', archive_dir)
        if not dry_run:
            os.makedirs(archive_dir)
    if logger is not None:
        logger.info("creating '%s' and adding '%s' to it", zip_filename, base_dir)
    if not dry_run:
        with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            arcname = os.path.normpath(base_dir)
            if root_dir is not None:
                base_dir = os.path.join(root_dir, base_dir)
            base_dir = os.path.normpath(base_dir)
            if arcname != os.curdir:
                zf.write(base_dir, arcname)
                if logger is not None:
                    logger.info("adding '%s'", base_dir)
            for dirpath, dirnames, filenames in os.walk(base_dir):
                arcdirpath = dirpath
                if root_dir is not None:
                    arcdirpath = os.path.relpath(arcdirpath, root_dir)
                arcdirpath = os.path.normpath(arcdirpath)
                for name in sorted(dirnames):
                    path = os.path.join(dirpath, name)
                    arcname = os.path.join(arcdirpath, name)
                    zf.write(path, arcname)
                    if logger is not None:
                        logger.info("adding '%s'", path)
                for name in filenames:
                    path = os.path.join(dirpath, name)
                    path = os.path.normpath(path)
                    if os.path.isfile(path):
                        arcname = os.path.join(arcdirpath, name)
                        zf.write(path, arcname)
                        if logger is not None:
                            logger.info("adding '%s'", path)
    if root_dir is not None:
        zip_filename = os.path.abspath(zip_filename)
    return zip_filename