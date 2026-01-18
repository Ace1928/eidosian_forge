import random
def ScaledColor(self, sr, sg, sb, er, eg, eb, num_steps, step):
    """Creates an interpolated rgb color between two rgb colors."""
    num_intervals = num_steps - 1
    dr = (er - sr) / num_intervals
    dg = (eg - sg) / num_intervals
    db = (eb - sb) / num_intervals
    r = sr + dr * step
    g = sg + dg * step
    b = sb + db * step
    return 'rgb(%i, %i, %i)' % (r, g, b)