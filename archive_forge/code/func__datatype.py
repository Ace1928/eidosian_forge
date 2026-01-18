import os.path as op
def _datatype(self):
    """Fetch the experiment datatype the pipeline applies to. If the
        pipeline does not cope with an specific datatype, function returns
        'All Datatypes'.
        """
    proj_pipes = Pipelines(self._project, self._intf)
    info = proj_pipes.info(self.id).pop()
    return info['Datatype']