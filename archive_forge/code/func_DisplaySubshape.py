def DisplaySubshape(viewer, shape, name, showSkelPts=True, color=(1, 0, 1)):
    import os
    import tempfile
    from rdkit import Geometry
    fName = tempfile.NamedTemporaryFile(suffix='.grd', delete=False).name
    Geometry.WriteGridToFile(shape.grid, fName)
    viewer.server.loadSurface(fName, name, '', 2.5)
    if showSkelPts:
        DisplaySubshapeSkeleton(viewer, shape, name, color)
    try:
        os.unlink(fName)
    except Exception:
        import time
        time.sleep(0.5)
        try:
            os.unlink(fName)
        except Exception:
            pass