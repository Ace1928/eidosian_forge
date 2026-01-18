import tree
import keras.src.backend
from keras.src.layers.layer import Layer
from keras.src.random.seed_generator import SeedGenerator
from keras.src.utils import backend_utils
from keras.src.utils import tracking
@tracking.no_automatic_dependency_tracking
def _get_seed_generator(self, backend=None):
    if backend is None or backend == keras.backend.backend():
        return self.generator
    if not hasattr(self, '_backend_generators'):
        self._backend_generators = {}
    if backend in self._backend_generators:
        return self._backend_generators[backend]
    seed_generator = SeedGenerator(self.seed, backend=self.backend)
    self._backend_generators[backend] = seed_generator
    return seed_generator