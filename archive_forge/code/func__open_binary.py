import importlib.resources
def _open_binary(pkg, res):
    return importlib.resources.files(pkg).joinpath(res).open('rb')