from matplotlib.contour import QuadContourSet
import matplotlib.path as mpath
import numpy as np
from cartopy.mpl import _MPL_38
class GeoContourSet(QuadContourSet):
    """
    A contourset designed to handle things like contour labels.

    """

    def clabel(self, *args, **kwargs):
        if not _MPL_38:
            for col in self.collections:
                paths = col.get_paths()
                data_t = self.axes.transData
                col_to_data = col.get_transform() - data_t
                new_paths = [col_to_data.transform_path(path) for path in paths]
                new_paths = [path for path in new_paths if path.vertices.size >= 1]
                col.set_transform(data_t)
                del paths[:]
                for path in new_paths:
                    if path.vertices.size == 0:
                        continue
                    codes = np.array(path.codes if path.codes is not None else [0])
                    moveto = codes == mpath.Path.MOVETO
                    if moveto.sum() <= 1:
                        paths.append(path)
                    else:
                        moveto[0] = False
                        split_locs = np.flatnonzero(moveto)
                        split_verts = np.split(path.vertices, split_locs)
                        split_codes = np.split(path.codes, split_locs)
                        for verts, codes in zip(split_verts, split_codes):
                            paths.append(mpath.Path(verts, codes))
        else:
            data_t = self.axes.transData
            col_to_data = self.get_transform() - data_t
            paths = self.get_paths()
            new_paths = [col_to_data.transform_path(path) for path in paths]
            self.set_paths(new_paths)
            self.set_transform(data_t)
        return super().clabel(*args, **kwargs)