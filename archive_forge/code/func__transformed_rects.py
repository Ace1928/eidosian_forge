import math
from itertools import islice
import numpy as np
import shapely
from shapely.affinity import affine_transform
def _transformed_rects():
    for dx, dy in edges:
        length = math.sqrt(dx ** 2 + dy ** 2)
        ux, uy = (dx / length, dy / length)
        vx, vy = (-uy, ux)
        transf_rect = affine_transform(hull, (ux, uy, vx, vy, 0, 0)).envelope
        yield (transf_rect, (ux, vx, uy, vy, 0, 0))