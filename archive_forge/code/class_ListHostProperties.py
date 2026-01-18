import logging
from blazarclient import command
from blazarclient import exception
class ListHostProperties(command.ListCommand):
    """List host properties."""
    resource = 'host'
    log = logging.getLogger(__name__ + '.ListHostProperties')
    list_columns = ['property', 'private', 'property_values']

    def args2body(self, parsed_args):
        params = {'detail': parsed_args.detail, 'all': parsed_args.all}
        if parsed_args.sort_by:
            if parsed_args.sort_by in self.list_columns:
                params['sort_by'] = parsed_args.sort_by
            else:
                msg = 'Invalid sort option %s' % parsed_args.sort_by
                raise exception.BlazarClientException(msg)
        return params

    def retrieve_list(self, parsed_args):
        """Retrieve a list of resources from Blazar server."""
        blazar_client = self.get_client()
        body = self.args2body(parsed_args)
        resource_manager = getattr(blazar_client, self.resource)
        data = resource_manager.list_properties(**body)
        return data

    def get_parser(self, prog_name):
        parser = super(ListHostProperties, self).get_parser(prog_name)
        parser.add_argument('--detail', action='store_true', help='Return properties with values and attributes.', default=False)
        parser.add_argument('--sort-by', metavar='<property_column>', help='column name used to sort result', default='property')
        parser.add_argument('--all', action='store_true', help='Return all properties, public and private.', default=False)
        return parser