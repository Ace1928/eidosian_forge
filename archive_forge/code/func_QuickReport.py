def QuickReport(conn, fileName, *args, **kwargs):
    title = 'Db Report'
    if 'title' in kwargs:
        title = kwargs['title']
        del kwargs['title']
    names = [x.upper() for x in conn.GetColumnNames()]
    try:
        smiCol = names.index('SMILES')
    except ValueError:
        try:
            smiCol = names.index('SMI')
        except ValueError:
            smiCol = -1
    if smiCol > -1:
        if hasCDX:
            tform = CDXImageTransformer(smiCol)
        elif 1:
            tform = ReportLabImageTransformer(smiCol)
        else:
            tform = CactvsImageTransformer(smiCol)
    else:
        tform = None
    kwargs['transform'] = tform
    tbl = conn.GetReportlabTable(*args, **kwargs)
    tbl.setStyle(platypus.TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('FONT', (0, 0), (-1, -1), 'Times-Roman', 8)]))
    if smiCol > -1 and tform:
        tbl._argW[smiCol] = tform.width * 1.2
    elements = [tbl]
    reportTemplate = PDFReport()
    reportTemplate.pageHeader = title
    doc = platypus.SimpleDocTemplate(fileName)
    doc.build(elements, onFirstPage=reportTemplate.onPage, onLaterPages=reportTemplate.onPage)