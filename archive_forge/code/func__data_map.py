from . import xpktools
def _data_map(labelline):
    labelList = labelline.split()
    datamap = {label: i for i, label in enumerate(labelList)}
    return datamap