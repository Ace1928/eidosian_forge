import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class TimeAwareRandomState(TimeAware):
    """
    Generic base class to enable time-dependent random
    streams. Although this class is the basis of all random numbergen
    classes, it is designed to be useful whenever time-dependent
    randomness is needed using param's notion of time. For instance,
    this class is used by the imagen package to define time-dependent,
    random distributions over 2D arrays.

    For generality, this class may use either the Random class from
    Python's random module or numpy.random.RandomState. Either of
    these random state objects may be used to generate numbers from
    any of several different random distributions (e.g. uniform,
    Gaussian). The latter offers the ability to generate
    multi-dimensional random arrays and more random distributions but
    requires numpy as a dependency.

    If declared time_dependent, the random state is fully determined
    by a hash value per call. The hash is initialized once with the
    object name and then per call using a tuple consisting of the time
    (via time_fn) and the global param.random_seed.  As a consequence,
    for a given name and fixed value of param.random_seed, the random
    values generated will be a fixed function of time.

    If the object name has not been set and time_dependent is True, a
    message is generated warning that the default object name is
    dependent on the order of instantiation.  To ensure that the
    random number stream will remain constant even if other objects
    are added or reordered in your file, supply a unique name
    explicitly when you construct the RandomDistribution object.
    """
    random_generator = param.Parameter(default=random.Random(c_size_t(hash((500, 500))).value), doc='\n        Random state used by the object. This may be an instance\n        of random.Random from the Python standard library or an\n        instance of numpy.random.RandomState.\n\n        This random state may be exclusively owned by the object or\n        may be shared by all instance of the same class. It is always\n        possible to give an object its own unique random state by\n        setting this parameter with a new random state instance.\n        ')
    __abstract = True

    def _initialize_random_state(self, seed=None, shared=True, name=None):
        """
        Initialization method to be called in the constructor of
        subclasses to initialize the random state correctly.

        If seed is None, there is no control over the random stream
        (no reproducibility of the stream).

        If shared is True (and not time-dependent), the random state
        is shared across all objects of the given class. This can be
        overridden per object by creating new random state to assign
        to the random_generator parameter.
        """
        if seed is None:
            seed = random.Random().randint(0, 1000000)
            suffix = ''
        else:
            suffix = str(seed)
        if self.time_dependent or not shared:
            self.random_generator = type(self.random_generator)(seed)
        if not shared:
            self.random_generator.seed(seed)
        if name is None:
            self._verify_constrained_hash()
        hash_name = name if name else self.name
        if not shared:
            hash_name += suffix
        self._hashfn = Hash(hash_name, input_count=2)
        if self.time_dependent:
            self._hash_and_seed()

    def _verify_constrained_hash(self):
        """
        Warn if the object name is not explicitly set.
        """
        changed_params = self.param.values(onlychanged=True)
        if self.time_dependent and 'name' not in changed_params:
            self.param.log(param.WARNING, 'Default object name used to set the seed: random values conditional on object instantiation order.')

    def _hash_and_seed(self):
        """
        To be called between blocks of random number generation. A
        'block' can be an unbounded sequence of random numbers so long
        as the time value (as returned by time_fn) is guaranteed not
        to change within the block. If this condition holds, each
        block of random numbers is time-dependent.

        Note: param.random_seed is assumed to be integer or rational.
        """
        hashval = self._hashfn(self.time_fn(), param.random_seed)
        self.random_generator.seed(hashval)