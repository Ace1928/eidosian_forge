import numpy as np
from ase.io.eps import EPS
class PNG(EPS):

    def write_header(self, fd):
        pass

    def _renderer(self, fd):
        from matplotlib.backends.backend_agg import RendererAgg
        dpi = 72
        return RendererAgg(self.w, self.h, dpi)

    def write_trailer(self, fd, renderer):
        import matplotlib.image
        buf = renderer.buffer_rgba()
        array = np.frombuffer(buf, dtype=np.uint8).reshape(int(self.h), int(self.w), 4)
        matplotlib.image.imsave(fd, array, format='png')