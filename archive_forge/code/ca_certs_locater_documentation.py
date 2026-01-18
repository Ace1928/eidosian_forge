import os
Locate the ca_certs.txt file.

  The httplib2 library will look for local ca_certs_locater module to override
  the default location for the ca_certs.txt file. We override it here to first
  try loading via resources, falling back to the traditional method if
  that fails.

  Returns:
    The file location returned as a string.
  