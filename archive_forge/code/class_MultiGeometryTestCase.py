import numpy as np
class MultiGeometryTestCase:

    def subgeom_access_test(self, cls, geoms):
        geom = cls(geoms)
        for t in test_int_types:
            for i, g in enumerate(geoms):
                assert geom.geoms[t(i)] == geoms[i]