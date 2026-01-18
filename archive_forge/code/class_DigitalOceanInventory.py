from __future__ import absolute_import, division, print_function
import argparse
import ast
import os
import re
import requests
import sys
from time import time
import json
class DigitalOceanInventory(object):

    def __init__(self):
        """Main execution path"""
        self.data = {}
        self.inventory = {}
        self.cache_path = '.'
        self.cache_max_age = 0
        self.use_private_network = False
        self.group_variables = {}
        self.droplets_tag_name = None
        self.read_settings()
        self.read_environment()
        self.read_cli_args()
        if not hasattr(self, 'api_token'):
            msg = 'Could not find values for DigitalOcean api_token. They must be specified via either ini file, command line argument (--api-token), or environment variables (DO_API_TOKEN)\n'
            sys.stderr.write(msg)
            sys.exit(-1)
        if self.args.env:
            print('DO_API_TOKEN=%s' % self.api_token)
            sys.exit(0)
        self.cache_filename = self.cache_path + '/ansible-digital_ocean.cache'
        self.cache_refreshed = False
        if self.is_cache_valid():
            self.load_from_cache()
            if len(self.data) == 0:
                if self.args.force_cache:
                    sys.stderr.write('Cache is empty and --force-cache was specified\n')
                    sys.exit(-1)
        self.manager = DoManager(self.api_token)
        if self.args.droplets:
            self.load_from_digital_ocean('droplets')
            json_data = {'droplets': self.data['droplets']}
        elif self.args.regions:
            self.load_from_digital_ocean('regions')
            json_data = {'regions': self.data['regions']}
        elif self.args.images:
            self.load_from_digital_ocean('images')
            json_data = {'images': self.data['images']}
        elif self.args.sizes:
            self.load_from_digital_ocean('sizes')
            json_data = {'sizes': self.data['sizes']}
        elif self.args.ssh_keys:
            self.load_from_digital_ocean('ssh_keys')
            json_data = {'ssh_keys': self.data['ssh_keys']}
        elif self.args.domains:
            self.load_from_digital_ocean('domains')
            json_data = {'domains': self.data['domains']}
        elif self.args.tags:
            self.load_from_digital_ocean('tags')
            json_data = {'tags': self.data['tags']}
        elif self.args.all:
            self.load_from_digital_ocean()
            json_data = self.data
        elif self.args.host:
            json_data = self.load_droplet_variables_for_host()
        else:
            self.load_from_digital_ocean('droplets')
            self.build_inventory()
            json_data = self.inventory
        if self.cache_refreshed:
            self.write_to_cache()
        if self.args.pretty:
            print(json.dumps(json_data, indent=2))
        else:
            print(json.dumps(json_data))

    def read_settings(self):
        """Reads the settings from the digital_ocean.ini file"""
        config = ConfigParser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'digital_ocean.ini')
        config.read(config_path)
        if config.has_option('digital_ocean', 'api_token'):
            self.api_token = config.get('digital_ocean', 'api_token')
        if config.has_option('digital_ocean', 'cache_path'):
            self.cache_path = config.get('digital_ocean', 'cache_path')
        if config.has_option('digital_ocean', 'cache_max_age'):
            self.cache_max_age = config.getint('digital_ocean', 'cache_max_age')
        if config.has_option('digital_ocean', 'use_private_network'):
            self.use_private_network = config.getboolean('digital_ocean', 'use_private_network')
        if config.has_option('digital_ocean', 'group_variables'):
            self.group_variables = ast.literal_eval(config.get('digital_ocean', 'group_variables'))
        if config.has_option('droplets', 'tag_name'):
            self.droplets_tag_name = config.get('droplets', 'tag_name')

    def read_environment(self):
        """Reads the settings from environment variables"""
        if os.getenv('DO_API_TOKEN'):
            self.api_token = os.getenv('DO_API_TOKEN')
        if os.getenv('DO_API_KEY'):
            self.api_token = os.getenv('DO_API_KEY')

    def read_cli_args(self):
        """Command line argument processing"""
        parser = argparse.ArgumentParser(description='Produce an Ansible Inventory file based on DigitalOcean credentials')
        parser.add_argument('--list', action='store_true', help='List all active Droplets as Ansible inventory (default: True)')
        parser.add_argument('--host', action='store', type=int, help='Get all Ansible inventory variables about the Droplet with the given ID')
        parser.add_argument('--all', action='store_true', help='List all DigitalOcean information as JSON')
        parser.add_argument('--droplets', '-d', action='store_true', help='List Droplets as JSON')
        parser.add_argument('--regions', action='store_true', help='List Regions as JSON')
        parser.add_argument('--images', action='store_true', help='List Images as JSON')
        parser.add_argument('--sizes', action='store_true', help='List Sizes as JSON')
        parser.add_argument('--ssh-keys', action='store_true', help='List SSH keys as JSON')
        parser.add_argument('--domains', action='store_true', help='List Domains as JSON')
        parser.add_argument('--tags', action='store_true', help='List Tags as JSON')
        parser.add_argument('--pretty', '-p', action='store_true', help='Pretty-print results')
        parser.add_argument('--cache-path', action='store', help='Path to the cache files (default: .)')
        parser.add_argument('--cache-max_age', action='store', help='Maximum age of the cached items (default: 0)')
        parser.add_argument('--force-cache', action='store_true', default=False, help='Only use data from the cache')
        parser.add_argument('--refresh-cache', '-r', action='store_true', default=False, help='Force refresh of cache by making API requests to DigitalOcean (default: False - use cache files)')
        parser.add_argument('--env', '-e', action='store_true', help='Display DO_API_TOKEN')
        parser.add_argument('--api-token', '-a', action='store', help='DigitalOcean API Token')
        self.args = parser.parse_args()
        if self.args.api_token:
            self.api_token = self.args.api_token
        if not self.args.droplets and (not self.args.regions) and (not self.args.images) and (not self.args.sizes) and (not self.args.ssh_keys) and (not self.args.domains) and (not self.args.tags) and (not self.args.all) and (not self.args.host):
            self.args.list = True

    def load_from_digital_ocean(self, resource=None):
        """Get JSON from DigitalOcean API"""
        if self.args.force_cache and os.path.isfile(self.cache_filename):
            return
        if self.is_cache_valid() and (not (resource == 'droplets' or resource is None)):
            return
        if self.args.refresh_cache:
            resource = None
        if resource == 'droplets' or resource is None:
            self.data['droplets'] = self.manager.all_active_droplets(tag_name=self.droplets_tag_name)
            self.cache_refreshed = True
        if resource == 'regions' or resource is None:
            self.data['regions'] = self.manager.all_regions()
            self.cache_refreshed = True
        if resource == 'images' or resource is None:
            self.data['images'] = self.manager.all_images()
            self.cache_refreshed = True
        if resource == 'sizes' or resource is None:
            self.data['sizes'] = self.manager.sizes()
            self.cache_refreshed = True
        if resource == 'ssh_keys' or resource is None:
            self.data['ssh_keys'] = self.manager.all_ssh_keys()
            self.cache_refreshed = True
        if resource == 'domains' or resource is None:
            self.data['domains'] = self.manager.all_domains()
            self.cache_refreshed = True
        if resource == 'tags' or resource is None:
            self.data['tags'] = self.manager.all_tags()
            self.cache_refreshed = True

    def add_inventory_group(self, key):
        """Method to create group dict"""
        host_dict = {'hosts': [], 'vars': {}}
        self.inventory[key] = host_dict
        return

    def add_host(self, group, host):
        """Helper method to reduce host duplication"""
        if group not in self.inventory:
            self.add_inventory_group(group)
        if host not in self.inventory[group]['hosts']:
            self.inventory[group]['hosts'].append(host)
        return

    def build_inventory(self):
        """Build Ansible inventory of droplets"""
        self.inventory = {'all': {'hosts': [], 'vars': self.group_variables}, '_meta': {'hostvars': {}}}
        for droplet in self.data['droplets']:
            for net in droplet['networks']['v4']:
                if net['type'] == 'public':
                    droplet['ip_address'] = net['ip_address']
                elif net['type'] == 'private':
                    droplet['private_ip_address'] = net['ip_address']
            host_indentifier = droplet['ip_address']
            if self.use_private_network and droplet['private_ip_address']:
                host_indentifier = droplet['private_ip_address']
            self.inventory['all']['hosts'].append(host_indentifier)
            self.add_host(droplet['id'], host_indentifier)
            self.add_host(droplet['name'], host_indentifier)
            for group in ('digital_ocean', 'region_' + droplet['region']['slug'], 'image_' + str(droplet['image']['id']), 'size_' + droplet['size']['slug'], 'distro_' + DigitalOceanInventory.to_safe(droplet['image']['distribution']), 'status_' + droplet['status']):
                self.add_host(group, host_indentifier)
            for group in (droplet['image']['slug'], droplet['image']['name']):
                if group:
                    image = 'image_' + DigitalOceanInventory.to_safe(group)
                    self.add_host(image, host_indentifier)
            if droplet['tags']:
                for tag in droplet['tags']:
                    self.add_host(tag, host_indentifier)
            info = self.do_namespace(droplet)
            self.inventory['_meta']['hostvars'][host_indentifier] = info

    def load_droplet_variables_for_host(self):
        """Generate a JSON response to a --host call"""
        droplet = self.manager.show_droplet(self.args.host)
        info = self.do_namespace(droplet)
        return info

    def is_cache_valid(self):
        """Determines if the cache files have expired, or if it is still valid"""
        if os.path.isfile(self.cache_filename):
            mod_time = os.path.getmtime(self.cache_filename)
            current_time = time()
            if mod_time + self.cache_max_age > current_time:
                return True
        return False

    def load_from_cache(self):
        """Reads the data from the cache file and assigns it to member variables as Python Objects"""
        try:
            with open(self.cache_filename, 'r') as cache:
                json_data = cache.read()
            data = json.loads(json_data)
        except IOError:
            data = {'data': {}, 'inventory': {}}
        self.data = data['data']
        self.inventory = data['inventory']

    def write_to_cache(self):
        """Writes data in JSON format to a file"""
        data = {'data': self.data, 'inventory': self.inventory}
        json_data = json.dumps(data, indent=2)
        with open(self.cache_filename, 'w') as cache:
            cache.write(json_data)

    @staticmethod
    def to_safe(word):
        """Converts 'bad' characters in a string to underscores so they can be used as Ansible groups"""
        return re.sub('[^A-Za-z0-9\\-.]', '_', word)

    @staticmethod
    def do_namespace(data):
        """Returns a copy of the dictionary with all the keys put in a 'do_' namespace"""
        info = {}
        for k, v in data.items():
            info['do_' + k] = v
        return info