import os
import os.path as op
import datetime
import string
import networkx as nx
from ...utils.filemanip import split_filename
from ..base import (
from .base import CFFBaseInterface, have_cfflib
class CFFConverter(CFFBaseInterface):
    """
    Creates a Connectome File Format (CFF) file from input networks, surfaces, volumes, tracts, etcetera....

    Example
    -------

    >>> import nipype.interfaces.cmtk as cmtk
    >>> cvt = cmtk.CFFConverter()
    >>> cvt.inputs.title = 'subject 1'
    >>> cvt.inputs.gifti_surfaces = ['lh.pial_converted.gii', 'rh.pial_converted.gii']
    >>> cvt.inputs.tract_files = ['streamlines.trk']
    >>> cvt.inputs.gpickled_networks = ['network0.gpickle']
    >>> cvt.run()                 # doctest: +SKIP
    """
    input_spec = CFFConverterInputSpec
    output_spec = CFFConverterOutputSpec

    def _run_interface(self, runtime):
        import cfflib as cf
        a = cf.connectome()
        if isdefined(self.inputs.title):
            a.connectome_meta.set_title(self.inputs.title)
        else:
            a.connectome_meta.set_title(self.inputs.out_file)
        if isdefined(self.inputs.creator):
            a.connectome_meta.set_creator(self.inputs.creator)
        else:
            a.connectome_meta.set_creator(os.getenv('USER'))
        if isdefined(self.inputs.email):
            a.connectome_meta.set_email(self.inputs.email)
        if isdefined(self.inputs.publisher):
            a.connectome_meta.set_publisher(self.inputs.publisher)
        if isdefined(self.inputs.license):
            a.connectome_meta.set_license(self.inputs.license)
        if isdefined(self.inputs.rights):
            a.connectome_meta.set_rights(self.inputs.rights)
        if isdefined(self.inputs.references):
            a.connectome_meta.set_references(self.inputs.references)
        if isdefined(self.inputs.relation):
            a.connectome_meta.set_relation(self.inputs.relation)
        if isdefined(self.inputs.species):
            a.connectome_meta.set_species(self.inputs.species)
        if isdefined(self.inputs.description):
            a.connectome_meta.set_description(self.inputs.description)
        a.connectome_meta.set_created(datetime.date.today())
        count = 0
        if isdefined(self.inputs.graphml_networks):
            for ntwk in self.inputs.graphml_networks:
                ntwk_name = 'Network {cnt}'.format(cnt=count)
                a.add_connectome_network_from_graphml(ntwk_name, ntwk)
                count += 1
        if isdefined(self.inputs.gpickled_networks):
            unpickled = []
            for ntwk in self.inputs.gpickled_networks:
                _, ntwk_name, _ = split_filename(ntwk)
                unpickled = _read_pickle(ntwk)
                cnet = cf.CNetwork(name=ntwk_name)
                cnet.set_with_nxgraph(unpickled)
                a.add_connectome_network(cnet)
                count += 1
        count = 0
        if isdefined(self.inputs.tract_files):
            for trk in self.inputs.tract_files:
                _, trk_name, _ = split_filename(trk)
                ctrack = cf.CTrack(trk_name, trk)
                a.add_connectome_track(ctrack)
                count += 1
        count = 0
        if isdefined(self.inputs.gifti_surfaces):
            for surf in self.inputs.gifti_surfaces:
                _, surf_name, _ = split_filename(surf)
                csurf = cf.CSurface.create_from_gifti('Surface %d - %s' % (count, surf_name), surf)
                csurf.fileformat = 'Gifti'
                csurf.dtype = 'Surfaceset'
                a.add_connectome_surface(csurf)
                count += 1
        count = 0
        if isdefined(self.inputs.gifti_labels):
            for label in self.inputs.gifti_labels:
                _, label_name, _ = split_filename(label)
                csurf = cf.CSurface.create_from_gifti('Surface Label %d - %s' % (count, label_name), label)
                csurf.fileformat = 'Gifti'
                csurf.dtype = 'Labels'
                a.add_connectome_surface(csurf)
                count += 1
        if isdefined(self.inputs.nifti_volumes):
            for vol in self.inputs.nifti_volumes:
                _, vol_name, _ = split_filename(vol)
                cvol = cf.CVolume.create_from_nifti(vol_name, vol)
                a.add_connectome_volume(cvol)
        if isdefined(self.inputs.script_files):
            for script in self.inputs.script_files:
                _, script_name, _ = split_filename(script)
                cscript = cf.CScript.create_from_file(script_name, script)
                a.add_connectome_script(cscript)
        if isdefined(self.inputs.data_files):
            for data in self.inputs.data_files:
                _, data_name, _ = split_filename(data)
                cda = cf.CData(name=data_name, src=data, fileformat='NumPy')
                if not string.find(data_name, 'lengths') == -1:
                    cda.dtype = 'FinalFiberLengthArray'
                if not string.find(data_name, 'endpoints') == -1:
                    cda.dtype = 'FiberEndpoints'
                if not string.find(data_name, 'labels') == -1:
                    cda.dtype = 'FinalFiberLabels'
                a.add_connectome_data(cda)
        a.print_summary()
        _, name, ext = split_filename(self.inputs.out_file)
        if not ext == '.cff':
            ext = '.cff'
        cf.save_to_cff(a, op.abspath(name + ext))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, name, ext = split_filename(self.inputs.out_file)
        if not ext == '.cff':
            ext = '.cff'
        outputs['connectome_file'] = op.abspath(name + ext)
        return outputs