import numpy as np
from scipy import stats
from . import utils
def _make_ellipse(mean, cov, ax, level=0.95, color=None):
    """Support function for scatter_ellipse."""
    from matplotlib.patches import Ellipse
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi
    v = 2 * np.sqrt(v * stats.chi2.ppf(level, 2))
    ell = Ellipse(mean[:2], v[0], v[1], angle=180 + angle, facecolor='none', edgecolor=color, lw=1.5)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)