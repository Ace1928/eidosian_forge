import logging
import re
from boto.vendored.regions.exceptions import NoRegionError
class BaseEndpointResolver(object):
    """Resolves regions and endpoints. Must be subclassed."""

    def construct_endpoint(self, service_name, region_name=None):
        """Resolves an endpoint for a service and region combination.

        :type service_name: string
        :param service_name: Name of the service to resolve an endpoint for
            (e.g., s3)

        :type region_name: string
        :param region_name: Region/endpoint name to resolve (e.g., us-east-1)
            if no region is provided, the first found partition-wide endpoint
            will be used if available.

        :rtype: dict
        :return: Returns a dict containing the following keys:
            - partition: (string, required) Resolved partition name
            - endpointName: (string, required) Resolved endpoint name
            - hostname: (string, required) Hostname to use for this endpoint
            - sslCommonName: (string) sslCommonName to use for this endpoint.
            - credentialScope: (dict) Signature version 4 credential scope
              - region: (string) region name override when signing.
              - service: (string) service name override when signing.
            - signatureVersions: (list<string>) A list of possible signature
              versions, including s3, v4, v2, and s3v4
            - protocols: (list<string>) A list of supported protocols
              (e.g., http, https)
            - ...: Other keys may be included as well based on the metadata
        """
        raise NotImplementedError

    def get_available_partitions(self):
        """Lists the partitions available to the endpoint resolver.

        :return: Returns a list of partition names (e.g., ["aws", "aws-cn"]).
        """
        raise NotImplementedError

    def get_available_endpoints(self, service_name, partition_name='aws', allow_non_regional=False):
        """Lists the endpoint names of a particular partition.

        :type service_name: string
        :param service_name: Name of a service to list endpoint for (e.g., s3)

        :type partition_name: string
        :param partition_name: Name of the partition to limit endpoints to.
            (e.g., aws for the public AWS endpoints, aws-cn for AWS China
            endpoints, aws-us-gov for AWS GovCloud (US) Endpoints, etc.

        :type allow_non_regional: bool
        :param allow_non_regional: Set to True to include endpoints that are
             not regional endpoints (e.g., s3-external-1,
             fips-us-gov-west-1, etc).
        :return: Returns a list of endpoint names (e.g., ["us-east-1"]).
        """
        raise NotImplementedError