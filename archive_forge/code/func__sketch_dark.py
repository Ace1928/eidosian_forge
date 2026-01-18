@_needs_mpl
def _sketch_dark():
    """Apply the sketch dark style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    almost_black = '#151515'
    plt.rcParams['figure.facecolor'] = almost_black
    plt.rcParams['savefig.facecolor'] = almost_black
    plt.rcParams['axes.facecolor'] = '#EBAAC1'
    plt.rcParams['patch.facecolor'] = '#B0B5DC'
    plt.rcParams['patch.edgecolor'] = 'white'
    plt.rcParams['patch.linewidth'] = 3.0
    plt.rcParams['patch.force_edgecolor'] = True
    plt.rcParams['lines.color'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['path.sketch'] = (1, 100, 2)