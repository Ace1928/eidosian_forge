from collections import OrderedDict
from copy import deepcopy
from pickle import dumps
import os
import getpass
import platform
from uuid import uuid1
import simplejson as json
import numpy as np
import prov.model as pm
from .. import get_info, logging, __version__
from .filemanip import md5, hashlib, hash_infile
class ProvStore(object):

    def __init__(self):
        self.g = pm.ProvDocument()
        self.g.add_namespace(foaf)
        self.g.add_namespace(dcterms)
        self.g.add_namespace(nipype_ns)
        self.g.add_namespace(niiri)

    def add_results(self, results, keep_provenance=False):
        if keep_provenance and results.provenance:
            self.g = deepcopy(results.provenance)
            return self.g
        runtime = results.runtime
        interface = results.interface
        inputs = results.inputs
        outputs = results.outputs
        classname = interface.__name__
        modulepath = '{0}.{1}'.format(interface.__module__, interface.__name__)
        activitytype = ''.join([i.capitalize() for i in modulepath.split('.')])
        a0_attrs = {nipype_ns['module']: interface.__module__, nipype_ns['interface']: classname, pm.PROV['type']: nipype_ns[activitytype], pm.PROV['label']: classname, nipype_ns['duration']: safe_encode(runtime.duration), nipype_ns['workingDirectory']: safe_encode(runtime.cwd), nipype_ns['returnCode']: safe_encode(runtime.returncode), nipype_ns['platform']: safe_encode(runtime.platform), nipype_ns['version']: safe_encode(runtime.version)}
        a0_attrs[foaf['host']] = pm.Literal(runtime.hostname, pm.XSD['anyURI'])
        try:
            a0_attrs.update({nipype_ns['command']: safe_encode(runtime.cmdline)})
            a0_attrs.update({nipype_ns['commandPath']: safe_encode(runtime.command_path)})
            a0_attrs.update({nipype_ns['dependencies']: safe_encode(runtime.dependencies)})
        except AttributeError:
            pass
        a0 = self.g.activity(get_id(), runtime.startTime, runtime.endTime, a0_attrs)
        id = get_id()
        env_collection = self.g.collection(id)
        env_collection.add_attributes({pm.PROV['type']: nipype_ns['Environment'], pm.PROV['label']: 'Environment'})
        self.g.used(a0, id)
        for idx, (key, val) in enumerate(sorted(runtime.environ.items())):
            if key not in PROV_ENVVARS:
                continue
            in_attr = {pm.PROV['label']: key, nipype_ns['environmentVariable']: key, pm.PROV['value']: safe_encode(val)}
            id = get_attr_id(in_attr)
            self.g.entity(id, in_attr)
            self.g.hadMember(env_collection, id)
        if inputs:
            id = get_id()
            input_collection = self.g.collection(id)
            input_collection.add_attributes({pm.PROV['type']: nipype_ns['Inputs'], pm.PROV['label']: 'Inputs'})
            for idx, (key, val) in enumerate(sorted(inputs.items())):
                in_entity = prov_encode(self.g, val).identifier
                self.g.hadMember(input_collection, in_entity)
                used_attr = {pm.PROV['label']: key, nipype_ns['inPort']: key}
                self.g.used(activity=a0, entity=in_entity, other_attributes=used_attr)
        if outputs:
            id = get_id()
            output_collection = self.g.collection(id)
            if not isinstance(outputs, dict):
                outputs = outputs.get_traitsfree()
            output_collection.add_attributes({pm.PROV['type']: nipype_ns['Outputs'], pm.PROV['label']: 'Outputs'})
            self.g.wasGeneratedBy(output_collection, a0)
            for idx, (key, val) in enumerate(sorted(outputs.items())):
                out_entity = prov_encode(self.g, val).identifier
                self.g.hadMember(output_collection, out_entity)
                gen_attr = {pm.PROV['label']: key, nipype_ns['outPort']: key}
                self.g.generation(out_entity, activity=a0, other_attributes=gen_attr)
        id = get_id()
        runtime_collection = self.g.collection(id)
        runtime_collection.add_attributes({pm.PROV['type']: nipype_ns['Runtime'], pm.PROV['label']: 'RuntimeInfo'})
        self.g.wasGeneratedBy(runtime_collection, a0)
        for key, value in sorted(runtime.items()):
            if not value:
                continue
            if key not in ['stdout', 'stderr', 'merged']:
                continue
            attr = {pm.PROV['label']: key, nipype_ns[key]: safe_encode(value)}
            id = get_id()
            self.g.entity(get_id(), attr)
            self.g.hadMember(runtime_collection, id)
        user_attr = {pm.PROV['type']: pm.PROV['Person'], pm.PROV['label']: getpass.getuser(), foaf['name']: safe_encode(getpass.getuser())}
        user_agent = self.g.agent(get_attr_id(user_attr), user_attr)
        agent_attr = {pm.PROV['type']: pm.PROV['SoftwareAgent'], pm.PROV['label']: 'Nipype', foaf['name']: safe_encode('Nipype'), nipype_ns['version']: __version__}
        for key, value in list(get_info().items()):
            agent_attr.update({nipype_ns[key]: safe_encode(value)})
        software_agent = self.g.agent(get_attr_id(agent_attr), agent_attr)
        self.g.wasAssociatedWith(a0, user_agent, None, None, {pm.PROV['hadRole']: nipype_ns['LoggedInUser']})
        self.g.wasAssociatedWith(a0, software_agent)
        return self.g

    def write_provenance(self, filename='provenance', format='all'):
        if format in ['provn', 'all']:
            with open(filename + '.provn', 'wt') as fp:
                fp.writelines(self.g.get_provn())
        try:
            if format in ['rdf', 'all']:
                if len(self.g.bundles) == 0:
                    rdf_format = 'turtle'
                    ext = '.ttl'
                else:
                    rdf_format = 'trig'
                    ext = '.trig'
                self.g.serialize(filename + ext, format='rdf', rdf_format=rdf_format)
            if format in ['jsonld']:
                self.g.serialize(filename + '.jsonld', format='rdf', rdf_format='json-ld', indent=4)
        except pm.serializers.DoNotExist:
            pass
        return self.g