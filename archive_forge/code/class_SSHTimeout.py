class SSHTimeout(IntegrationException):
    message = 'Connection to the %(host)s via SSH timed out.\nUser: %(user)s, Password: %(password)s'