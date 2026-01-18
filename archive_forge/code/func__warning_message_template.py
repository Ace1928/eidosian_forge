import warnings
def _warning_message_template(self):
    msg = '%(func_name)s is deprecated'
    if self.last_supported_version is not None:
        msg += ' since (not including) % s' % self.last_supported_version
    if self.will_be_missing_in is not None:
        msg += ', it will be missing in %s' % self.will_be_missing_in
    if self.issue is not None:
        if self.issues_url is not None:
            msg += self.issues_url(self.issue)
        else:
            msg += ' (see issue %s)' % self.issue
    if self.use_instead is not None:
        try:
            msg += '. Use %s instead' % self.use_instead.__name__
        except AttributeError:
            msg += '. Use %s instead' % self.use_instead
    return msg + '.'