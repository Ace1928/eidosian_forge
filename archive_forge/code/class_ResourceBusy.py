class ResourceBusy(PathError):
    _fmt = 'Device or resource busy: "%(path)s"%(extra)s'