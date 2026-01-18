@_needs_mpl
def _solarized_light():
    """Apply the solarized light style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    plt.rcParams['savefig.facecolor'] = '#fdf6e3'
    plt.rcParams['figure.facecolor'] = '#fdf6e3'
    plt.rcParams['axes.facecolor'] = '#eee8d5'
    plt.rcParams['patch.edgecolor'] = '#93a1a1'
    plt.rcParams['patch.linewidth'] = 3.0
    plt.rcParams['patch.facecolor'] = '#eee8d5'
    plt.rcParams['lines.color'] = '#657b83'
    plt.rcParams['text.color'] = '#586e75'
    plt.rcParams['patch.force_edgecolor'] = True
    plt.rcParams['path.sketch'] = None