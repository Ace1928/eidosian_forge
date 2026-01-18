import os
import hashlib
import pickle
import time
import shutil
import glob
from ..interfaces.base import BaseInterface
from ..pipeline.engine import Node
from ..pipeline.engine.utils import modify_paths
class PipeFunc(object):
    """Callable interface to nipype.interface objects

    Use this to wrap nipype.interface object and call them
    specifying their input with keyword arguments::

        fsl_merge = PipeFunc(fsl.Merge, base_dir='.')
        out = fsl_merge(in_files=files, dimension='t')
    """

    def __init__(self, interface, base_dir, callback=None):
        """

        Parameters
        ===========
        interface: a nipype interface class
            The interface class to wrap
        base_dir: a string
            The directory in which the computation will be
            stored
        callback: a callable
            An optional callable called each time after the function
            is called.
        """
        if not (isinstance(interface, type) and issubclass(interface, BaseInterface)):
            raise ValueError('the interface argument should be a nipype interface class, but %s (type %s) was passed.' % (interface, type(interface)))
        self.interface = interface
        base_dir = os.path.abspath(base_dir)
        if not os.path.exists(base_dir) and os.path.isdir(base_dir):
            raise ValueError('base_dir should be an existing directory')
        self.base_dir = base_dir
        doc = '%s\n%s' % (self.interface.__doc__, self.interface.help(returnhelp=True))
        self.__doc__ = doc
        self.callback = callback

    def __call__(self, **kwargs):
        kwargs = modify_paths(kwargs, relative=False)
        interface = self.interface()
        interface.inputs.trait_set(**kwargs)
        inputs = interface.inputs.get_hashval()
        hasher = hashlib.new('md5')
        hasher.update(pickle.dumps(inputs))
        dir_name = '%s-%s' % (interface.__class__.__module__.replace('.', '-'), interface.__class__.__name__)
        job_name = hasher.hexdigest()
        node = Node(interface, name=job_name)
        node.base_dir = os.path.join(self.base_dir, dir_name)
        cwd = os.getcwd()
        try:
            out = node.run()
        finally:
            os.chdir(cwd)
        if self.callback is not None:
            self.callback(dir_name, job_name)
        return out

    def __repr__(self):
        return '{}({}.{}), base_dir={})'.format(self.__class__.__name__, self.interface.__module__, self.interface.__name__, self.base_dir)