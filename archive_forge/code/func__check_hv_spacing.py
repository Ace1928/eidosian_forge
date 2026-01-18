import collections
def _check_hv_spacing(dimsize, spacing, name, dimvarname, dimname):
    if spacing < 0 or spacing > 1:
        raise ValueError('%s spacing must be between 0 and 1.' % (name,))
    if dimsize <= 1:
        return
    max_spacing = 1.0 / float(dimsize - 1)
    if spacing > max_spacing:
        raise ValueError('{name} spacing cannot be greater than (1 / ({dimvarname} - 1)) = {max_spacing:f}.\nThe resulting plot would have {dimsize} {dimname} ({dimvarname}={dimsize}).'.format(dimvarname=dimvarname, name=name, dimname=dimname, max_spacing=max_spacing, dimsize=dimsize))