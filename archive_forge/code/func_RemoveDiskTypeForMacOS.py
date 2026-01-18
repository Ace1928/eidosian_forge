def RemoveDiskTypeForMacOS(ref, args, request):
    del ref, args
    if request.workerPool.hostOs is not None and request.workerPool.hostOs.startswith('macos-'):
        request.workerPool.workerConfig.diskType = None
    return request