import io
import numpy as np
import os
import pandas as pd
import warnings
from math import log, floor
from numbers import Number
from plotly import optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.exceptions import PlotlyError
import plotly.graph_objs as go
def _create_us_counties_df(st_to_state_name_dict, state_to_st_dict):
    abs_dir_path = os.path.realpath(_plotly_geo.__file__)
    abs_plotly_geo_path = os.path.dirname(abs_dir_path)
    abs_package_data_dir_path = os.path.join(abs_plotly_geo_path, 'package_data')
    shape_pre2010 = 'gz_2010_us_050_00_500k.shp'
    shape_pre2010 = os.path.join(abs_package_data_dir_path, shape_pre2010)
    df_shape_pre2010 = gp.read_file(shape_pre2010)
    df_shape_pre2010['FIPS'] = df_shape_pre2010['STATE'] + df_shape_pre2010['COUNTY']
    df_shape_pre2010['FIPS'] = pd.to_numeric(df_shape_pre2010['FIPS'])
    states_path = 'cb_2016_us_state_500k.shp'
    states_path = os.path.join(abs_package_data_dir_path, states_path)
    df_state = gp.read_file(states_path)
    df_state = df_state[['STATEFP', 'NAME', 'geometry']]
    df_state = df_state.rename(columns={'NAME': 'STATE_NAME'})
    filenames = ['cb_2016_us_county_500k.dbf', 'cb_2016_us_county_500k.shp', 'cb_2016_us_county_500k.shx']
    for j in range(len(filenames)):
        filenames[j] = os.path.join(abs_package_data_dir_path, filenames[j])
    dbf = io.open(filenames[0], 'rb')
    shp = io.open(filenames[1], 'rb')
    shx = io.open(filenames[2], 'rb')
    r = shapefile.Reader(shp=shp, shx=shx, dbf=dbf)
    attributes, geometry = ([], [])
    field_names = [field[0] for field in r.fields[1:]]
    for row in r.shapeRecords():
        geometry.append(shapely.geometry.shape(row.shape.__geo_interface__))
        attributes.append(dict(zip(field_names, row.record)))
    gdf = gp.GeoDataFrame(data=attributes, geometry=geometry)
    gdf['FIPS'] = gdf['STATEFP'] + gdf['COUNTYFP']
    gdf['FIPS'] = pd.to_numeric(gdf['FIPS'])
    f = 46113
    singlerow = pd.DataFrame([[st_to_state_name_dict['SD'], 'SD', df_shape_pre2010[df_shape_pre2010['FIPS'] == f]['geometry'].iloc[0], df_shape_pre2010[df_shape_pre2010['FIPS'] == f]['FIPS'].iloc[0], '46', 'Shannon']], columns=['State', 'ST', 'geometry', 'FIPS', 'STATEFP', 'NAME'], index=[max(gdf.index) + 1])
    gdf = pd.concat([gdf, singlerow], sort=True)
    f = 51515
    singlerow = pd.DataFrame([[st_to_state_name_dict['VA'], 'VA', df_shape_pre2010[df_shape_pre2010['FIPS'] == f]['geometry'].iloc[0], df_shape_pre2010[df_shape_pre2010['FIPS'] == f]['FIPS'].iloc[0], '51', 'Bedford City']], columns=['State', 'ST', 'geometry', 'FIPS', 'STATEFP', 'NAME'], index=[max(gdf.index) + 1])
    gdf = pd.concat([gdf, singlerow], sort=True)
    f = 2270
    singlerow = pd.DataFrame([[st_to_state_name_dict['AK'], 'AK', df_shape_pre2010[df_shape_pre2010['FIPS'] == f]['geometry'].iloc[0], df_shape_pre2010[df_shape_pre2010['FIPS'] == f]['FIPS'].iloc[0], '02', 'Wade Hampton']], columns=['State', 'ST', 'geometry', 'FIPS', 'STATEFP', 'NAME'], index=[max(gdf.index) + 1])
    gdf = pd.concat([gdf, singlerow], sort=True)
    row_2198 = gdf[gdf['FIPS'] == 2198]
    row_2198.index = [max(gdf.index) + 1]
    row_2198.loc[row_2198.index[0], 'FIPS'] = 2201
    row_2198.loc[row_2198.index[0], 'STATEFP'] = '02'
    gdf = pd.concat([gdf, row_2198], sort=True)
    row_2105 = gdf[gdf['FIPS'] == 2105]
    row_2105.index = [max(gdf.index) + 1]
    row_2105.loc[row_2105.index[0], 'FIPS'] = 2232
    row_2105.loc[row_2105.index[0], 'STATEFP'] = '02'
    gdf = pd.concat([gdf, row_2105], sort=True)
    gdf = gdf.rename(columns={'NAME': 'COUNTY_NAME'})
    gdf_reduced = gdf[['FIPS', 'STATEFP', 'COUNTY_NAME', 'geometry']]
    gdf_statefp = gdf_reduced.merge(df_state[['STATEFP', 'STATE_NAME']], on='STATEFP')
    ST = []
    for n in gdf_statefp['STATE_NAME']:
        ST.append(state_to_st_dict[n])
    gdf_statefp['ST'] = ST
    return (gdf_statefp, df_state)